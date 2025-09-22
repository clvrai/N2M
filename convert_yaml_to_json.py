import yaml
import json
import os
from typing import Dict, List, Any

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
                    episode_data["theta"]
                ]
            },
            "file_path": f"pcl/{int(episode_id) - 1}.pcd",
            "task_name": "02_05",
            "meta_info": {
                "layout": 2,
                "style": 5,
                "se2_origin": [
                    4.049999999676611,
                    -0.7999999999978153,
                    1.5707962925663528
                ],
                "furniture_name": "cab_main_main_group",
                "furniture_pos": [
                    4.05,
                    -0.2,
                    1.85
                ],
                "seed": 123000
            }
        }
        output["episodes"].append(episode)
    
    return output

def main():
    # Read YAML file
    input_file = "input.yaml"
    output_file = "output.json"
    
    try:
        with open(input_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Convert the data
        json_data = convert_yaml_to_json(yaml_data)
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        print(f"Successfully converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    main() 