import yaml
import json
import os
from typing import Dict, List, Any
import argparse
import shutil
import numpy as np

def convert_yaml_to_json(yaml_data: Dict) -> Dict:
    """
    Convert YAML data to the specified JSON format.
    
    Args:
        yaml_data (Dict): Input YAML data
        
    Returns:
        Dict: Converted JSON data
    """
    # Initialize the output structure
    output = {
        "meta": {
            "trial_count": len(yaml_data),
            "success_count": sum(1 for episode in yaml_data.values() if episode.get("success", False)),
            "success_rate": 0.0
        },
        "episodes": []
    }
    
    # Calculate success rate
    output["meta"]["success_rate"] = output["meta"]["success_count"] / output["meta"]["trial_count"]
    
    # Convert each episode
    for episode_id, episode_data in yaml_data.items():
        episode = {
            "id": int(episode_id) - 1,  # Convert to 0-based indexing
            "is_success": 1.0 if episode_data.get("success", False) else 0.0,
            "pose": {
                "se2": [
                    episode_data["x"],
                    episode_data["y"],
                    episode_data["theta"],
                    episode_data["torso_1"]
                ]
            },
            "file_path": f"pcl/{int(episode_id)}.pcd",
            "meta_info": {
                "se2_origin": [
                    episode_data["x"],
                    episode_data["y"],
                    episode_data["theta"]
                ],
                "furniture_name": "lamp",
                "furniture_pos": [
                    episode_data["x"] + 0.5 * np.cos(episode_data["theta"]),
                    episode_data["y"] + 0.5 * np.sin(episode_data["theta"]),
                    episode_data["theta"]
                ],
            }
        }
        output["episodes"].append(episode)
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="datasets/rollouts/lamp_120/rollouts_rainbow")
    parser.add_argument("--output", type=str, default="datasets/rollouts/lamp_120/rollouts_sir")
    args = parser.parse_args()
    # Read YAML file
    input_meta = os.path.join(args.input, "meta_data.yaml")
    output_meta = os.path.join(args.output, "meta.json")

    os.makedirs(args.output, exist_ok=True)
    
    try:
        with open(input_meta, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Convert the data
        json_data = convert_yaml_to_json(yaml_data)
        
        # Write to JSON file
        with open(output_meta, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        print(f"Successfully converted {input_meta} to {output_meta}")

        # copy pcd files
        pcd_dir = os.path.join(args.input, "pcl")
        output_pcd_dir = os.path.join(args.output, "pcl")
        os.makedirs(output_pcd_dir, exist_ok=True)
        for file in os.listdir(pcd_dir):
            shutil.copy(os.path.join(pcd_dir, file), os.path.join(output_pcd_dir, file))
        
        print(f"Successfully copied {pcd_dir} to {output_pcd_dir}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main() 