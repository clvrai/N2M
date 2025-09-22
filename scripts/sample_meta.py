import json
import argparse
import random
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir")
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset_path, 'rollout/meta_aug.json')

    sample_number = [20, 40, 60, 80, 100, 200, 300]

    with open(meta_path, "r") as f:
        meta = json.load(f)

    for sample_num in sample_number:
        sample_idx = random.sample(range(300), sample_num)
        print(len(sample_idx))

        new_meta = {
            "meta": meta['meta'],
            "episodes": []
        }
        for episode in meta['episodes']:
            if episode['id'] in sample_idx:
                new_meta['episodes'].append(episode)
        
        output_path = os.path.join(args.dataset_path, f'rollout/meta_aug_{sample_num:03d}.json')
        with open(output_path, "w") as f:
            json.dump(new_meta, f, indent=4)
