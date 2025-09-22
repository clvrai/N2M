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
    ids = []
    id_div_5 = []
    for episode in meta['episodes']:
        if episode['is_success'] == 1.0:
            if int(episode['id'] / 5) % 2 == 1 or (int(episode['id'] / 5) in id_div_5 and episode['id'] not in ids):
                continue
            new_meta['episodes'].append(episode)
            if int(episode['id'] / 5) not in id_div_5:
                id_div_5.append(int(episode['id'] / 5))
                ids.append(episode['id'])
    
    output_path = os.path.join(args.dataset_path, f'rollout/meta_aug_half_positive_one.json')
    with open(output_path, "w") as f:
        json.dump(new_meta, f, indent=4)
