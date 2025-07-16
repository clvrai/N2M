import torch
import torch.nn as nn
import torch.nn.functional as F
from nav2man.model.pointbert.point_encoder import PointTransformer
import yaml
import os
import numpy as np

class N2Mnet(nn.Module):
    def __init__(self, config):
        """
        Initialize the SIR Predictor model
        This is a model that predicts the distribution of SIR points from the point cloud
        
        Args:
            config: Config dict from yaml file
            num_gaussians: Number of Gaussian components in the mixture
            output_dim: Dimension of the output (e.g., 3 for xyz coordinates)
        """
        super().__init__()
        self.encoder_config = config['encoder']
        self.decoder_config = config['decoder']

        self.num_gaussians = self.decoder_config['num_gaussians']
        self.output_dim = self.decoder_config['output_dim']
        
        # Load PointBERT config
        if self.encoder_config['name'] == 'PointBERT':
            self.encoder = PointTransformer(self.encoder_config['config'])

            # Get the output dimension of PointBERT
            encoder_output_dim = self.encoder_config['config']['trans_dim'] * 2 # *2 because of max pooling
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_config['name']}")
        
        # Load pretrained weights if specified
        if 'ckpt' in self.encoder_config:
            if not os.path.exists(self.encoder_config['ckpt']):
                raise FileNotFoundError(f"Checkpoint file not found: {self.encoder_config['ckpt']}")
            print(f"Loading model weights from {self.encoder_config['ckpt']}")
            self.encoder.load_checkpoint(self.encoder_config['ckpt'])
        
        # Freeze encoder weights
        freeze = self.encoder_config['freeze'] if 'freeze' in self.encoder_config else False
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # MLP layers for predicting GMM parameters
        # For each Gaussian component, we need:
        # - mean (output_dim)
        # - full covariance matrix (output_dim * output_dim)
        # - mixing coefficient (1)
        decoder_output_dim = self.output_dim + (self.output_dim * self.output_dim) + 1
        
        if self.decoder_config['name'] == 'mlp':
            if 'num_settings' in self.decoder_config:
                decoder_input_dim = encoder_output_dim + self.decoder_config['num_settings']
            else:
                decoder_input_dim = encoder_output_dim

            if 'layers' in self.decoder_config:
                layers = self.decoder_config['layers']
            else:
                layers = [512, 256]

            decoder_layers = []
            prev_dim = decoder_input_dim
            for layer_dim in layers:
                decoder_layers.extend([
                    nn.Linear(prev_dim, layer_dim),
                    nn.ReLU(),
                    nn.Dropout(self.decoder_config['config']['dropout'])
                ])
                prev_dim = layer_dim
            
            decoder_layers.append(nn.Linear(prev_dim, self.num_gaussians * decoder_output_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder_config['name']}")

        if 'ckpt' in config:
            if not os.path.exists(config['ckpt']):
                raise FileNotFoundError(f"Checkpoint file not found: {config['ckpt']}")
            print(f"Loading model weights from {config['ckpt']}")
            self.load_state_dict(torch.load(config['ckpt'])['model_state_dict'])
            
    def _construct_covariance_matrices(self, sigma_params):
        """
        Construct positive semidefinite covariance matrices using matrix exponential
        """
        B, K, D, _ = sigma_params.shape
        
        # Make the matrix symmetric
        sigma_params = 0.5 * (sigma_params + sigma_params.transpose(-2, -1))
        
        # Add small diagonal term for numerical stability
        eye = torch.eye(D, device=sigma_params.device).unsqueeze(0).unsqueeze(0)
        sigma_params = sigma_params + 1e-6 * eye
        
        # Compute matrix exponential to ensure positive definiteness
        covs = torch.matrix_exp(sigma_params)
        
        return covs
        
    def forward(self, point_cloud, task_idx=None):
        """
        Forward pass of the SIR Predictor model
        input pointcloud is processed as in SIRDataset
        
        Args:
            point_cloud: Input point cloud (B, N, C)
            task_idx: Task index (B, )
        Returns:
            means: Mean vectors for each Gaussian component (B, num_gaussians, output_dim)
            covs: Full covariance matrices for each Gaussian component (B, num_gaussians, output_dim, output_dim)
            weights: Mixing coefficients for each Gaussian component (B, num_gaussians)
        """
        # Get features from PointBERT
        features = self.encoder(point_cloud)  # (B, 1, C)
        features = features.squeeze(1)  # (B, C)

        if task_idx is not None:
            task_one_hot = F.one_hot(task_idx, num_classes=self.decoder_config['num_settings'])
            task_features = task_one_hot.to(torch.float32)
            features = torch.cat([features, task_features], dim=-1) # (B, C + task_num)
        
        # Get GMM parameters from MLP
        gmm_params = self.decoder(features)  # (B, num_gaussians * total_params_per_component)
        
        # Reshape parameters
        batch_size = gmm_params.size(0)
        
        # Split parameters into means, covariance elements, and weights
        means = gmm_params[:, :self.num_gaussians * self.output_dim].view(batch_size, self.num_gaussians, self.output_dim)
        
        # Get covariance matrix parameters
        start_idx = self.num_gaussians * self.output_dim
        end_idx = start_idx + self.num_gaussians * self.output_dim * self.output_dim
        sigma_params = gmm_params[:, start_idx:end_idx].view(
            batch_size, self.num_gaussians, self.output_dim, self.output_dim
        )
        
        # Construct positive semidefinite covariance matrices
        covs = self._construct_covariance_matrices(sigma_params)
        
        # Get mixing coefficients and apply softmax
        weights = gmm_params[:, -self.num_gaussians:].view(batch_size, self.num_gaussians) # (B, K)
        weights = torch.softmax(weights, dim=-1) # (B, K)
        
        return means, covs, weights
    
    def sample(self, point_cloud, task_idx=None, num_samples=1000):
        """
        Sample from the SIR Predictor model
        input pointcloud is processed as in SIRDataset
        
        Args:
            point_cloud: Input point cloud (B, N, C)
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated samples (B, num_samples, output_dim)
        """
        means, covs, weights = self.forward(point_cloud, task_idx)
        batch_size = means.size(0)
        
        # Initialize output tensor
        samples = torch.zeros(batch_size, num_samples, self.output_dim, device=means.device)
        
        # Sample from each Gaussian component
        for b in range(batch_size):
            # Sample component indices based on weights
            component_indices = torch.multinomial(weights[b], num_samples, replacement=True)
            
            # Sample from each component
            for i in range(self.num_gaussians):
                # Get number of samples for this component
                num_component_samples = (component_indices == i).sum().item()
                if num_component_samples > 0:
                    # Sample from multivariate normal with full covariance
                    dist = torch.distributions.MultivariateNormal(
                        means[b, i],
                        covs[b, i]
                    )
                    samples[b, component_indices == i] = dist.sample((num_component_samples,))
        
        return samples, means, covs, weights