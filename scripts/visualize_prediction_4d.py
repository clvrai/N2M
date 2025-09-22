import argparse
import torch
import json
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcl_path", type=str, required=True)
    parser.add_argument("--sir_config", type=str, required=True)
    args = parser.parse_args()

    from nav2man.model.SIRPredictor import SIRPredictor
    from nav2man.utils.point_cloud import fix_point_cloud_size
    from nav2man.utils.visualizer import save_gmm_visualization_xythetaz
    
    with open(args.sir_config, "r") as f:
        SIR_config = json.load(f)
    
    pcd = o3d.io.read_point_cloud(args.pcl_path)
    pcl_numpy = np.asarray(pcd.points)
    if pcd.has_colors():
        pcl_numpy = np.concatenate([pcl_numpy, np.asarray(pcd.colors)], axis=1)
    pcl_input = fix_point_cloud_size(pcl_numpy, SIR_config['dataset']['pointnum'])
    pcl_input = torch.from_numpy(pcl_input)[None, ...].cuda().float()

    model_config = SIR_config["model"]
    SIR_predictor = SIRPredictor(config=model_config)
    SIR_predictor.eval()
    SIR_predictor.cuda()
    with torch.no_grad():
        means, covs, weights = SIR_predictor(pcl_input)
        means = means[0].cpu().numpy()
        covs = covs[0].cpu().numpy()
        weights = weights[0].cpu().numpy()

    save_gmm_visualization_xythetaz(
        pcl_numpy,
        None,
        None,
        means,
        covs,
        weights,
        f"vis_pcl.ply",
        interval=0.05, 
        area=[-10, 10, -10, 10, -10, 10], 
        threshold=0.5)