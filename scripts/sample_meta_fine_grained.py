import json
import argparse
import random
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir")
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset_path, 'rollout/meta_robot_centric.json')

    sample_number = [2, 4, 8, 12, 16, 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    positive_id = [episode['id'] for episode in meta['episodes'] if episode['is_success'] == 1]
    negative_id = [episode['id'] for episode in meta['episodes'] if episode['is_success'] == 0]
    
    for sample_num in sample_number:
        print(f"sampling {sample_num} episodes...")
        episodes = []
        if sample_num <= 8:
            sample_positive_id = positive_id[:sample_num]
            episodes.extend([episode for episode in meta['episodes'] if episode['id'] in sample_positive_id])
            print(f"sampled {sample_num} positive episodes")
        else:
            positive_num = int(sample_num * 0.8)
            negative_num = sample_num - positive_num

            if positive_num > len(positive_id):
                print("not enough positive episodes skipping...")
                continue
            if negative_num > len(negative_id):
                print("not enough negative episodes skipping...")
                continue

            sample_positive_id = positive_id[:positive_num]
            sample_negative_id = negative_id[:negative_num]
            episodes.extend([episode for episode in meta['episodes'] if episode['id'] in sample_positive_id])
            episodes.extend([episode for episode in meta['episodes'] if episode['id'] in sample_negative_id])
            print(f"sampled {positive_num} positive episodes, {negative_num} negative episodes")
        
        new_meta = {
            "meta": meta['meta'],
            "episodes": episodes
        }
        
        output_path = os.path.join(args.dataset_path, f'rollout/meta_robot_centric_{sample_num:03d}.json')
        with open(output_path, "w") as f:
            json.dump(new_meta, f, indent=4)
