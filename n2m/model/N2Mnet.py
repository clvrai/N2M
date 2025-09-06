import torch
import torch.nn as nn
import torch.nn.functional as F
from n2m.model.pointbert.point_encoder import PointTransformer
import os
from typing import Tuple, Dict


class N2Mnet(nn.Module):
    """
    N2Mnet predicts a Gaussian Mixture Model (GMM) from an input point cloud.
    """

    def __init__(self, config: Dict):
        """
        Initializes the N2Mnet model.

        Args:
            config (Dict): A configuration dictionary containing model, encoder,
                           and decoder settings.
        """
        super().__init__()
        self.encoder_config = config['encoder']
        self.decoder_config = config['decoder']

        self.num_gaussians = self.decoder_config['num_gaussians']
        self.output_dim = self.decoder_config['output_dim']

        encoder_output_dim = self._build_encoder()
        self._build_decoder(encoder_output_dim)

        if 'ckpt' in config and config['ckpt']:
            if not os.path.exists(config['ckpt']):
                raise FileNotFoundError(f"Checkpoint file not found: {config['ckpt']}")
            print(f"Loading model weights from {config['ckpt']}")
            self.load_state_dict(torch.load(config['ckpt'])['model_state_dict'])

    def _build_encoder(self) -> int:
        """
        Builds the point cloud encoder based on the configuration.

        Returns:
            int: The output dimension of the encoder.
        """
        if self.encoder_config['name'] == 'PointBERT':
            self.encoder = PointTransformer(self.encoder_config['config'])
            # Output dimension is doubled due to max and mean pooling in PointTransformer
            encoder_output_dim = self.encoder_config['config']['trans_dim'] * 2
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_config['name']}")

        # Load pretrained encoder weights if specified
        if 'ckpt' in self.encoder_config and self.encoder_config['ckpt']:
            ckpt_path = self.encoder_config['ckpt']
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")
            print(f"Loading encoder weights from {ckpt_path}")
            self.encoder.load_checkpoint(ckpt_path)

        # Freeze encoder weights if specified
        if self.encoder_config.get('freeze', False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        return encoder_output_dim

    def _build_decoder(self, decoder_input_dim: int):
        """
        Builds the GMM parameter decoder based on the configuration.

        Args:
            decoder_input_dim (int): The input dimension for the decoder.
        """
        # Each Gaussian component requires parameters for mean, covariance, and mixing weight
        gmm_params_per_gaussian = self.output_dim + (self.output_dim ** 2) + 1
        decoder_output_dim = self.num_gaussians * gmm_params_per_gaussian

        if self.decoder_config['name'] == 'mlp':
            layers = self.decoder_config.get('layers', [512, 256])

            decoder_layers = []
            prev_dim = decoder_input_dim
            for layer_dim in layers:
                decoder_layers.extend([
                    nn.Linear(prev_dim, layer_dim),
                    nn.ReLU(),
                    nn.Dropout(self.decoder_config['config']['dropout'])
                ])
                prev_dim = layer_dim

            decoder_layers.append(nn.Linear(prev_dim, decoder_output_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder_config['name']}")

    def _construct_covariance_matrices(self, sigma_params: torch.Tensor) -> torch.Tensor:
        """
        Constructs positive semidefinite covariance matrices from raw network outputs.

        This method ensures symmetry and positive definiteness using the matrix
        exponential. A small identity matrix is added for numerical stability before
        the exponential.

        Args:
            sigma_params (torch.Tensor): Raw covariance parameters from the decoder,
                                       with shape (B, K, D, D), where B is batch
                                       size, K is the number of Gaussians, and D
                                       is the output dimension.

        Returns:
            torch.Tensor: Valid, positive semidefinite covariance matrices of shape
                        (B, K, D, D).
        """
        B, K, D, _ = sigma_params.shape

        # Enforce symmetry for the input to matrix exponential
        sigma_params = 0.5 * (sigma_params + sigma_params.transpose(-2, -1))

        # Add a small diagonal epsilon for numerical stability
        eye = torch.eye(D, device=sigma_params.device).unsqueeze(0).unsqueeze(0)
        sigma_params = sigma_params + 1e-6 * eye

        # The matrix exponential of a symmetric matrix is symmetric positive definite
        covs = torch.matrix_exp(sigma_params)

        return covs

    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass to predict GMM parameters from a point cloud.

        Args:
            point_cloud (torch.Tensor): Input point cloud of shape (B, N, C), where B
                                      is batch size, N is the number of points,
                                      and C is feature dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - means (torch.Tensor): GMM means (B, K, D).
                - covs (torch.Tensor): GMM covariance matrices (B, K, D, D).
                - weights (torch.Tensor): GMM mixing weights (B, K).
        """
        # Encode point cloud to a global feature vector
        features, _ = self.encoder(point_cloud)
        features = features.squeeze(1)  # (B, encoder_output_dim)

        # Decode features into a flat tensor of GMM parameters
        gmm_params = self.decoder(features)

        # Reshape the flat tensor to extract means, covariance parameters, and weights
        batch_size = gmm_params.size(0)

        # Extract means
        means_end = self.num_gaussians * self.output_dim
        means = gmm_params[:, :means_end].view(
            batch_size, self.num_gaussians, self.output_dim
        )

        # Extract raw covariance parameters
        cov_end = means_end + self.num_gaussians * (self.output_dim ** 2)
        sigma_params = gmm_params[:, means_end:cov_end].view(
            batch_size, self.num_gaussians, self.output_dim, self.output_dim
        )
        covs = self._construct_covariance_matrices(sigma_params)

        # Extract and normalize mixing weights
        weights = gmm_params[:, -self.num_gaussians:].view(batch_size, self.num_gaussians)
        weights = torch.softmax(weights, dim=-1)

        return means, covs, weights

    def sample(self, point_cloud: torch.Tensor, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples points from the predicted GMM for a given point cloud.

        Args:
            point_cloud (torch.Tensor): Input point cloud of shape (B, N, C).
            num_samples (int): Number of points to sample for each item in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of:
                - samples (torch.Tensor): Sampled points (B, num_samples, D).
                - means (torch.Tensor): GMM means (B, K, D).
                - covs (torch.Tensor): GMM covariances (B, K, D, D).
                - weights (torch.Tensor): GMM weights (B, K).
        """
        means, covs, weights = self.forward(point_cloud)
        batch_size = means.size(0)

        samples = torch.zeros(batch_size, num_samples, self.output_dim, device=means.device)

        # Process each item in the batch independently
        for b in range(batch_size):
            # 1. Sample component indices based on the mixture weights
            component_indices = torch.multinomial(weights[b], num_samples, replacement=True)

            # 2. Sample from the corresponding Gaussian for each chosen component
            for i in range(self.num_gaussians):
                # Find which samples belong to the current component
                mask = (component_indices == i)
                num_component_samples = mask.sum().item()

                if num_component_samples > 0:
                    dist = torch.distributions.MultivariateNormal(
                        loc=means[b, i],
                        covariance_matrix=covs[b, i]
                    )
                    samples[b, mask] = dist.sample((num_component_samples,))

        return samples, means, covs, weights