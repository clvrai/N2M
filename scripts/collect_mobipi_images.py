#!/usr/bin/env python3
"""Collect images for Mobipi 3DGS training.

TODO: To be implemented.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Collect Mobipi training data.
    
    TODO: Implement multi-view image collection for 3DGS reconstruction.
    
    Example usage:
        python scripts/collect_mobipi_images.py env=PnPCounterToCab benchmark=data_collection benchmark.collect_for=mobipi
    """
    raise NotImplementedError(
        "Mobipi image collection not yet implemented. "
        "This requires collecting multi-view RGB images and camera poses "
        "for 3D Gaussian Splatting reconstruction."
    )


if __name__ == "__main__":
    main()
