import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import open3d as o3d

from n2m.utils.point_cloud import apply_augmentations, fix_point_cloud_size

def make_data_module(config):
    """
    Make training dataset N2M.

    input(config):
        data_path: str
        anno_path: str
        use_color: bool
        pointnum: int

    return:
        train_dataset: Dataset
        val_dataset: Dataset
    """
    if isinstance(config['anno_path'], str):
        anno_path = os.path.join(config['dataset_path'], config['anno_path'])
        with open(anno_path, 'r') as f:
            anno = json.load(f)['episodes']
        train_num = int(len(anno) * config['train_val_ratio'])
        random.shuffle(anno)
        anno_train = anno[:train_num]
        anno_val = anno[train_num:]

        print(f"Train set size: {len(anno_train)}")
        print(f"Val set size: {len(anno_val)}")
        
        train_dataset = N2MDataset(
            config=config,
            anno=anno_train,
        )
        val_dataset = N2MDataset(
            config=config,
            anno=anno_val,
        )

        return train_dataset, val_dataset
    else:
        raise ValueError(f"Invalid config: {config}")

class N2MDataset(Dataset):
    """Dataset for N2M"""
    def __init__(self, config, anno):
        self.anno = anno
        self.pointnum = config['pointnum']
        self.config = config

        self.settings = config['settings'] if 'settings' in config else None

        if 'dataset_path' in config:
            self.dataset_path = config['dataset_path']
        else:
            raise ValueError(f"Invalid config: {config}")

    def _load_point_cloud(self, file_path):
        if os.path.exists(file_path):
            pcd = o3d.io.read_point_cloud(file_path)
            if 'augmentations' in self.config and 'hpr' in self.config['augmentations']:
                pcd = self._apply_hpr(pcd, self.config['augmentations']['hpr'])
            point_cloud = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            point_cloud = np.concatenate([point_cloud, colors], axis=1)
            return point_cloud
                    
        raise FileNotFoundError(f"No point cloud file found for object {file_path}")
    
    def __len__(self):
        """
        Return number of samples in the dataset
        """
        return len(self.anno)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        """
        data = self.anno[index]
        file_path = os.path.join(self.dataset_path, data['file_path'])

        # Load point cloud and target point
        point_cloud = self._load_point_cloud(file_path)
        target_point = np.array(data['pose'], dtype=np.float32)
        label = 1

        # Ensure point cloud has consistent size
        point_cloud = fix_point_cloud_size(point_cloud, self.pointnum)

        # Apply augmentations
        if 'augmentations' in self.config:
            point_cloud, target_point = apply_augmentations(point_cloud, target_point, self.config['augmentations'])

        # Convert to torch tensors
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        target_point = torch.from_numpy(target_point)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'point_cloud': point_cloud,
            'target_point': target_point,
            'label': label,
        }
