import json
import argparse
import random
import os
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/rollouts/PnPCounterToCab_BCtransformer_rollouts/PnPCounterToCab_BCtransformer_rollout_scene1/rollouts_sir")
    parser.add_argument("--layouts", type=int, nargs="+", default=None)
    parser.add_argument("--styles", type=int, nargs="+", default=None)
    parser.add_argument("--meta_name", type=str, default="meta")
    parser.add_argument("--sample_num", type=int, default=10)
    args = parser.parse_args()

    meta_path = os.path.join(args.dataset_path, f'rollout/{args.meta_name}.json')

    with open(meta_path, "r") as f:
        meta = json.load(f)

    num_samples = np.zeros((10, 10))
    ids = {}
    new_meta = {
        "meta": meta['meta'],
        "episodes": []
    }
    for episode in meta['episodes']:
        if (args.layouts is None or episode['meta_info']['layout'] in args.layouts) and (args.styles is None or episode['meta_info']['style'] in args.styles) and num_samples[episode['meta_info']['layout'], episode['meta_info']['style']] <= args.sample_num:
            if f"{episode['meta_info']['layout']}_{episode['meta_info']['style']}" not in ids or episode['id'] not in ids[f"{episode['meta_info']['layout']}_{episode['meta_info']['style']}"]:
                if num_samples[episode['meta_info']['layout'], episode['meta_info']['style']] == args.sample_num:
                    continue
                num_samples[episode['meta_info']['layout'], episode['meta_info']['style']] += 1
                if f"{episode['meta_info']['layout']}_{episode['meta_info']['style']}" not in ids:
                    ids[f"{episode['meta_info']['layout']}_{episode['meta_info']['style']}"] = []
                ids[f"{episode['meta_info']['layout']}_{episode['meta_info']['style']}"].append(episode['id'])
            new_meta['episodes'].append(episode)
    
    layout_name = ''.join(map(str, args.layouts)) if args.layouts is not None else None
    style_name = ''.join(map(str, args.styles)) if args.styles is not None else None
    out_meta_name = f'{args.meta_name}_layout_{layout_name}_style_{style_name}' if layout_name is not None and style_name is not None else f'{args.meta_name}_layout_{layout_name}' if layout_name is not None else f'{args.meta_name}_style_{style_name}'
    output_path = os.path.join(args.dataset_path, f'rollout/{out_meta_name}.json')
    with open(output_path, "w") as f:
        json.dump(new_meta, f, indent=4)

    for layout in range(10):
        for style in range(10):
            if num_samples[layout, style] > 0:
                print(f'layout {layout} style {style} has {num_samples[layout, style]} samples')
                print(ids[f"{layout}_{style}"])
    