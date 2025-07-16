import torch
import os
import numpy as np
from torch.distributions import MultivariateNormal

from nav2man.utils.point_cloud import fix_point_cloud_size
from nav2man.utils.visualizer import save_gmm_visualization_se2
from nav2man.utils.point_cloud import translate_se2_point_cloud, translate_se2_target

def predict_SIR_target_point(
    SIR_predictor,
    SIR_config,
    pc_numpy,
    target_helper,
    SIR_sample_num,
    robot_centric,
    abs_base_se2,
    task_name,
    max_trial=100,
    save_dir=None,
    id=None,
):
    pcl_input = fix_point_cloud_size(pc_numpy, SIR_config['dataset']['pointnum'])
    if robot_centric:
        print ("abs_base_se2", abs_base_se2)
        pcl_input = translate_se2_point_cloud(pcl_input, [-abs_base_se2[0], -abs_base_se2[1], abs_base_se2[2]])
    pcl_input = torch.from_numpy(pcl_input)[None, ...].cuda().float()

    task_idx = None
    if 'settings' in SIR_config['dataset']:
        task_idx = torch.tensor([SIR_config['dataset']['settings'].index(task_name)]).cuda()

    target_SIR_prediction = None
    SIR_predictor.eval()
    with torch.no_grad():
        finished = False
        trial_count = 0
        while not finished:
            if trial_count > max_trial:
                print(f"Failed to find a valid target after {max_trial} trials")
                break

            samples, means, covs, weights = SIR_predictor.sample(pcl_input, task_idx, SIR_sample_num)

            num_modes = means.shape[1]
            mvns = [MultivariateNormal(means[0, i], covs[0, i]) for i in range(num_modes)]
            log_probs = torch.stack([mvn.log_prob(samples[0]) for mvn in mvns])  # shape: [num_modes, SIR_sample_num]
            log_weights = torch.log(weights[0] + 1e-8).unsqueeze(1)  # shape: [num_modes, 1]
            weighted_log_probs = log_probs + log_weights
            gaussian_probabilities = torch.logsumexp(weighted_log_probs, dim=0)  # shape: [SIR_sample_num]
            sorted_indices = torch.argsort(gaussian_probabilities, descending=True)
            samples = samples[0, sorted_indices]

            # convert to numpy
            means = means[0].cpu().numpy()
            covs = covs[0].cpu().numpy()
            weights = weights[0].cpu().numpy()
            samples = samples.cpu().numpy()

            # First set target_SIR_prediction to the mean with highest probability
            target_SIR_prediction = means[0]
            max_prob = 0
            for i in range(num_modes):
                # Calculate probability density at the mean using the covariance and weight
                mvn = MultivariateNormal(torch.from_numpy(means[i]), torch.from_numpy(covs[i]))
                prob = mvn.log_prob(torch.from_numpy(means[i])).exp().item() * weights[i]
                if prob > max_prob:
                    max_prob = prob
                    target_SIR_prediction = means[i]
            
            if save_dir is not None:
                prediction_folder = os.path.join(save_dir, "prediction")
                os.makedirs(prediction_folder, exist_ok=True)
                save_gmm_visualization_se2(
                    pcl_input[0].cpu().numpy(),
                    target_SIR_prediction,
                    1,
                    means,
                    covs,
                    weights,
                    os.path.join(prediction_folder, f"{id}.ply"),
                )

            if robot_centric:
                global_target_SIR_prediction = translate_se2_target(target_SIR_prediction.copy(), [abs_base_se2[0], abs_base_se2[1], -abs_base_se2[2]])
            else:
                global_target_SIR_prediction = target_SIR_prediction

            if not target_helper.check_collision(global_target_SIR_prediction): # target_helper is based on the relative position
                print("target_SIR_prediction is collided with the furniture, try again")
            elif not target_helper.check_boundary(global_target_SIR_prediction): # target_helper is based on the relative position
                print("target_SIR_prediction is out of the boundary, try again")
            else:
                print("target_SIR_prediction is valid, using mean with highest probability")
                finished = True
                break
            

            # If the mean with highest probability is not valid, try with samples
            for i in range(SIR_sample_num):
                target_SIR_prediction = samples[i]

                if save_dir is not None:
                    prediction_folder = os.path.join(save_dir, "prediction")
                    os.makedirs(prediction_folder, exist_ok=True)
                    save_gmm_visualization_se2(
                        pcl_input[0].cpu().numpy(),
                        target_SIR_prediction,
                        1,
                        means,
                        covs,
                        weights,
                        os.path.join(prediction_folder, f"{id}.ply"),
                    )

                if robot_centric:
                    global_target_SIR_prediction = translate_se2_target(target_SIR_prediction.copy(), [abs_base_se2[0], abs_base_se2[1], -abs_base_se2[2]])
                else:
                    global_target_SIR_prediction = target_SIR_prediction

                if not target_helper.check_collision(global_target_SIR_prediction): # target_helper is based on the relative position
                    print("target_SIR_prediction is collided with the furniture, try again")
                elif not target_helper.check_boundary(global_target_SIR_prediction): # target_helper is based on the relative position
                    print("target_SIR_prediction is out of the boundary, try again")
                else:
                    print("target_SIR_prediction is valid, using sample")
                    finished = True
                    break

            trial_count += 1

    return global_target_SIR_prediction, target_SIR_prediction, means, covs, weights, pcl_input[0].cpu().numpy(), finished