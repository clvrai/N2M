import numpy as np

from torch.distributions import MultivariateNormal

from n2m.models.N2Mnet import N2Mnet
from n2m.utils.point_cloud import fix_point_cloud_size
from n2m.utils.sample_utils import CollisionChecker

class N2Mmodule:
    """
    N2M module class that encapsulates the N2M model.
    This includes collision checking point cloud pre-processing.
    """
    def __init__(self, config):
        self.n2mnet_config = config['n2mnet']
        self.preprocess_config = config['preprocess']
        self.postprocess_config = config['postprocess']

        self.model = N2Mnet(self.n2mnet_config)
        self.collision_checker = CollisionChecker(self.postprocess_config['collision_checker'])

    def predict(self, point_cloud):
        """
        Predicts the preferable initial pose.

        Inputs:
            point_cloud (numpy.array): Input point cloud of shape (N, 3). The point cloud should be captured from robot centric camera and should have robot base at the origin.
        
        Outputs:
            preferable initial_pose (numpy.array): Predicted preferable initial pose of shape (3) for SE(2) (4) for SE(2) + z.
            prediction success (Boolean): There might not be a valid sample within the sample number provided. return the validity of the sample
        """
        self.model.eval()

        # preprocess(downsmaple) point cloud to match the specified pointnum
        pointnum = self.preprocess_config['pointnum']
        orig_point_cloud = point_cloud.copy()
        point_cloud = fix_point_cloud_size(point_cloud, pointnum)

        # load point cloud for collision checking
        self.collision_checker.set_pcd(orig_point_cloud)

        # get predictions
        with torch.no_grad():
            point_cloud_tensor = torch.tensor(input_point_cloud, dtype=torch.float32).unsqueeze(0)  # Add batch dimension (1, N, 3)
            num_samples = self.postprocess_config['num_samples']
            samples, means, covs, weights = self.model.sample(point_cloud_tensor, num_samples=num_samples)

        # first check mean's collision. If it doesn't collide, return mean value
        if self.collision_checker.check_collision(means[0]):
            return means[0], True

        # sort predictions in the order of predicted probability
        num_modes = weights.shape[1]
        mvns = [MultivariateNormal(means[0, i], covs[0, i]) for i in range(num_modes)]
        log_probs = torch.stack([mvn.log_prob(samples[0]) for mvn in mvns])  # shape: [num_modes, num_samples]
        log_weights = torch.log(weights[0] + 1e-8).unsqueeze(1) # shape: [num_modes, 1]
        weighted_log_probs = log_probs + log_weights
        gaussian_probabilities = torch.logsumexp(weighted_log_probs, dim=0) # shape: [num_samples]
        sorted_indices = torch.argsort(gaussian_probabilities, descending=True)
        samples = samples[0, sorted_indices]

        # check collision for each samples and return the non-colliding sample with the highest probability
        for i in range(num_samples):
            sample = samples[0, i]
            if self.collision_checker.check_collision(sample):
                print("Valid initial pose found: ", sample)
                return sample, True
            else:
                print("Invalid pose, trying again: ", sample)

        # prediction fail. return False for validity
        return means[0], False