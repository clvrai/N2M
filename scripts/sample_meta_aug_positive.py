import json
import argparse
import random
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir")
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset_path, 'rollout/meta_aug.json')

    with open(meta_path, "r") as f:
        meta = json.load(f)

    new_meta = {
        "meta": meta['meta'],
        "episodes": []
    }
    for episode in meta['episodes']:
        if episode['is_success'] == 1.0:
            new_meta['episodes'].append(episode)
    
    output_path = os.path.join(args.dataset_path, f'rollout/meta_aug_positive.json')
    with open(output_path, "w") as f:
        json.dump(new_meta, f, indent=4)
