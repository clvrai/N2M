import json
import argparse
import random
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default="")
    args = parser.parse_args()

    for meta_path in os.listdir(args.meta_dir):
        with open(os.path.join(args.meta_dir, meta_path), "r") as f:
            meta = json.load(f)

        positive_id = [episode['id'] for episode in meta['episodes'] if episode['is_success'] == 1]
        negative_id = [episode['id'] for episode in meta['episodes'] if episode['is_success'] == 0]

        print(f"{meta_path} positive: {len(positive_id)}, negative: {len(negative_id)}")
        print(f"positive ratio: {len(positive_id) / (len(positive_id) + len(negative_id))}")
