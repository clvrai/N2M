import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir/rollout/meta.json")
    args = parser.parse_args()

    with open(args.meta_path, "r") as f:
        meta = json.load(f)

    new_meta = {
        "meta": meta['meta'],
        "episodes": []
    }
    for episode in meta['episodes']:
        episode['file_path'] = episode['file_path'].split('/rollout/')[1]
        new_meta['episodes'].append(episode)

    with open(args.meta_path, "w") as f:
        json.dump(new_meta, f, indent=4)
