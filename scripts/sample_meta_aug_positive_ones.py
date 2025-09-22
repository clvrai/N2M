import json
import argparse
import random
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir")
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset_path, 'rollout/meta_aug_positive_robot_centric_origin.json')

    with open(meta_path, "r") as f:
        meta = json.load(f)

    id_set = set()
    for episode in meta['episodes']:
        id_set.add(episode['id'])
    id_list = list(id_set)

    sample_number = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    for sample_num in sample_number:
        if len(id_set) < sample_num:
            print(f"Not enough episodes to sample {sample_num} episodes")
            exit(0)
        sample_idx = id_list[:sample_num]
        print(f"Sampling {sample_num} episodes: {sample_idx}")

        new_meta = {
            "meta": meta['meta'],
            "episodes": []
        }
        for episode in meta['episodes']:
            if episode['id'] in sample_idx and episode["file_path"].split("/")[-1].split("_")[-1].split(".pcd")[0] == "0":
                new_meta['episodes'].append(episode)
        
        output_path = os.path.join(args.dataset_path, f'rollout/meta_aug_positive_robot_centric_origin_ones_{sample_num:03d}.json')
        with open(output_path, "w") as f:
            json.dump(new_meta, f, indent=4)
