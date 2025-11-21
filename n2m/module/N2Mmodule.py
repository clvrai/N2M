import numpy as np
import torch
import os

from torch.distributions import MultivariateNormal

from n2m.model.N2Mnet import N2Mnet
from n2m.utils.point_cloud import fix_point_cloud_size
from n2m.utils.sample_utils import CollisionChecker
from n2m.utils.visualizer import save_gmm_visualization_se2

class N2Mmodule:
    """
    N2M module class that encapsulates the N2M model.
    This includes collision checking point cloud pre-processing.
    """
    def __init__(self, config):
        self.n2mnet_config = config['n2mnet']
        self.preprocess_config = config['preprocess']
        self.postprocess_config = config['postprocess']
        self.ckpt_path = config['ckpt']

        self.model = N2Mnet(self.n2mnet_config)
        
        # Load checkpoint if provided
        if self.ckpt_path is not None:
            self.model.load_state_dict(torch.load(self.ckpt_path)['model_state_dict'])
            print(f"[N2Mmodule] Checkpoint loaded successfully")
        else:
            print("[N2Mmodule] Warning: No checkpoint provided, using random weights!")
        
        # self.collision_checker = CollisionChecker(self.postprocess_config['collision_checker'])

    def predict(self, point_cloud, collision_checker):
        """
        Predicts the preferable initial pose.

        Inputs:
            point_cloud (numpy.array): Input point cloud of shape (N, 3). The point cloud should be captured from robot centric camera and should have robot base at the origin.
            collision_checker: Collision checker instance
        
        Outputs:
            preferable initial_pose (numpy.array): Predicted preferable initial pose of shape (3) for SE(2) (4) for SE(2) + z.
            prediction success (Boolean): There might not be a valid sample within the sample number provided. return the validity of the sample
            extra_info (dict): Dictionary containing timing information
        """
        import time
        
        self.model.eval()

        # preprocess(downsmaple) point cloud to match the specified pointnum
        pointnum = self.preprocess_config['pointnum']
        # Handle both Open3D PointCloud and numpy array inputs
        import numpy as np
        if hasattr(point_cloud, "points"):  # Open3D PointCloud
            orig_point_cloud = point_cloud
            # Extract both xyz and rgb for the model (expects N, 6)
            points = np.asarray(point_cloud.points).astype(np.float32)
            colors = np.asarray(point_cloud.colors).astype(np.float32)
            point_cloud_np = np.concatenate([points, colors], axis=1)
        else:  # numpy array
            point_cloud_np = point_cloud.copy()
            orig_point_cloud = point_cloud_np
        point_cloud = fix_point_cloud_size(point_cloud_np, pointnum)

        # get predictions - time this section
        model_inference_start = time.time()
        with torch.no_grad():
            point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32, device=self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device).unsqueeze(0)  # Add batch dimension (1, N, 3)
            num_samples = self.postprocess_config['num_samples']
            samples, means, covs, weights = self.model.sample(point_cloud_tensor, num_samples=num_samples)
        model_inference_time = time.time() - model_inference_start

        # Convert to numpy - start filtering time
        collision_checking_start = time.time()
        means_np = means[0].cpu().numpy()  # (num_gaussians, 3)
        weights_np = weights[0].cpu().numpy()  # (num_gaussians,)
        
        # first check mean with highest weight for collision
        best_mean_idx = np.argmax(weights_np)
        best_mean = means_np[best_mean_idx]  # (3,)
        
        if collision_checker.check_collision(best_mean):
            print("Valid initial pose found: ", best_mean)
            collision_checking_time = time.time() - collision_checking_start
            extra_info = {
                'model_inference_time': model_inference_time,
                'collision_checking_time': collision_checking_time,
                'prediction_validity': True
            }
            save_gmm_visualization_se2(
                point_cloud = point_cloud,
                target_se2 = best_mean,
                label = 1,
                means = means_np,
                covs = covs[0].cpu().numpy(),
                weights = weights_np,
                output_path = "./debug4/valid_pose.ply"
            )
            return best_mean, extra_info

        # sort predictions in the order of predicted probability
        num_modes = weights.shape[1]
        mvns = [MultivariateNormal(means[0, i], covs[0, i]) for i in range(num_modes)]
        log_probs = torch.stack([mvn.log_prob(samples[0]) for mvn in mvns])  # shape: [num_modes, num_samples]
        log_weights = torch.log(weights[0] + 1e-8).unsqueeze(1) # shape: [num_modes, 1]
        weighted_log_probs = log_probs + log_weights
        gaussian_probabilities = torch.logsumexp(weighted_log_probs, dim=0) # shape: [num_samples]
        sorted_indices = torch.argsort(gaussian_probabilities, descending=True)
        samples = samples[0, sorted_indices].cpu().numpy()

        # check collision for each samples and return the non-colliding sample with the highest probability
        for i in range(num_samples):
            sample = samples[i]
            if collision_checker.check_collision(sample):
                print("Valid initial pose found: ", sample)
                collision_checking_time = time.time() - collision_checking_start
                extra_info = {
                    'model_inference_time': model_inference_time,
                    'collision_checking_time': collision_checking_time,
                    'prediction_validity': True
                }
                os.makedirs("./debug4", exist_ok=True)
                save_gmm_visualization_se2(
                    point_cloud = point_cloud,
                    target_se2 = sample,
                    label = 1,
                    means = means_np,
                    covs = covs[0].cpu().numpy(),
                    weights = weights_np,
                    output_path = "./debug4/valid_pose.ply"
                )
                return sample, extra_info
            else:
                print("Invalid pose, trying again: ", sample)

        # prediction fail. return False for validity
        collision_checking_time = time.time() - collision_checking_start
        extra_info = {
            'model_inference_time': model_inference_time,
            'collision_checking_time': collision_checking_time,
            'prediction_validity': False
        }
        os.makedirs("./debug4", exist_ok=True)
        save_gmm_visualization_se2(
            point_cloud = point_cloud,
            target_se2 = best_mean,
            label = 1,
            means = means_np,
            covs = covs[0].cpu().numpy(),
            weights = weights_np,
            output_path = "./debug4/valid_pose.ply"
        )
        return best_mean, extra_info