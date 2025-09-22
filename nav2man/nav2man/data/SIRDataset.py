import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import open3d as o3d

from nav2man.utils.point_cloud import apply_augmentations, fix_point_cloud_size

def make_SIR_data_module(config, PTR=None):
    """
    Make training dataset for SIR predictor.

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
        
        train_dataset = SIRDataset(
            config=config,
            anno=anno_train,
        )
        val_dataset = SIRDataset(
            config=config,
            anno=anno_val,
        )

        return train_dataset, val_dataset
    
    elif isinstance(config['anno_path'], dict) and PTR is not None:
        assert 'pretrain' in config['anno_path'] and 'finetune' in config['anno_path'], "anno_path must contain 'pretrain' and 'finetune' keys"

        anno_pretrain_path = os.path.join(config['pretrain_path'], config['anno_path']['pretrain'])
        anno_finetune_path = os.path.join(config['finetune_path'], config['anno_path']['finetune'])

        with open(anno_pretrain_path, 'r') as f:
            anno_pretrain = json.load(f)['episodes']
        with open(anno_finetune_path, 'r') as f:
            anno_finetune = json.load(f)['episodes']

        random.shuffle(anno_pretrain)
        random.shuffle(anno_finetune)
            
        if PTR == 'pretrain':
            train_num = int(len(anno_pretrain) * config['train_val_ratio'])
            anno_train = anno_pretrain[:train_num]
            anno_val = anno_pretrain[train_num:]

            print(f"Preparing data for PTR pretrain")
            print(f"Train set size: {len(anno_train)}")
            print(f"Val set size: {len(anno_val)}")

            train_dataset = SIRDataset(
                config=config,
                anno=anno_train,
            )
            val_dataset = SIRDataset(
                config=config,
                anno=anno_val,
            )
            return train_dataset, val_dataset
        
        elif PTR == 'finetune':
            # We don't split finetune set into train and val since the dataset is too small
            train_num = int(len(anno_finetune) * config['train_val_ratio'])
            anno_finetune_train = anno_finetune[:train_num]
            anno_finetune_val = anno_finetune[train_num:]

            print(f"Preparing data for PTR finetune")
            print(f"Train set size: {len(anno_finetune_train)}")
            print(f"Val set size: {len(anno_finetune_val)}")

            train_dataset = SIRDataset(
                config=config,
                anno_pretrain=anno_pretrain,
                anno=anno_finetune_train,
            )
            val_dataset = SIRDataset(
                config=config,
                anno=anno_finetune_val,
                mode='PTR_finetune_val'
            )
            return train_dataset, val_dataset
        
        else:
            raise ValueError(f"Invalid PTR: {PTR}")
    else:
        raise ValueError(f"Invalid config: {config}")

class SIRDataset(Dataset):
    """Dataset for SIR predictor."""
    def __init__(self, config, anno, anno_pretrain=None, mode=None):
        self.anno = anno
        self.anno_pretrain = anno_pretrain
        self.pointnum = config['pointnum']
        self.config = config

        self.settings = config['settings'] if 'settings' in config else None
        self.mix_finetune = True if self.anno_pretrain is not None else False
        if self.mix_finetune:
            self.finetune_ratio = config['finetune_ratio']

        if 'dataset_path' in config:
            self.dataset_path = config['dataset_path']
        elif mode is not None and mode == 'PTR_finetune_val':
            self.dataset_path = config['finetune_path']
        elif not self.mix_finetune:
            self.dataset_path = config['pretrain_path']
        else:
            self.dataset_path = config['finetune_path']
            self.pretrain_path = config['pretrain_path']

        # Load the data list from JSON
        # print(f"Loading anno file from {self.anno_path}.")
        # with open(self.config['anno_path'], "r") as json_file:
        #     self.list_data_dict = json.load(json_file)['episodes']
        # if 'PTR' in self.config:
        #     with open(self.config['PTR']['finetune_anno_path'], "r") as json_file:
        #         self.finetune_list_data_dict = json.load(json_file)['episodes']

        
        # print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")
        # random.seed(42)
        # random.shuffle(self.list_data_dict)

        # Split train and val with 9:1 ratio if specified
        # if self.config['split_train_val']:
        #     if self.split == 'train':
        #         self.list_data_dict = self.list_data_dict[:int(self.config['split_ratio'] * len(self.list_data_dict))]
        #         print(f"Train set size: {len(self.list_data_dict)}")
        #     else:
        #         self.list_data_dict = self.list_data_dict[int(self.config['split_ratio'] * len(self.list_data_dict)):]
        #         print(f"Val set size: {len(self.list_data_dict)}")

        # Debug mode: use subset of data
        # if self.config['data_debug_num'] > 0:
        #     self.list_data_dict = self.list_data_dict[:self.config['data_debug_num']]
        #     print('Debug mode, using: ' + ' '.join([data['object_id'] for data in self.list_data_dict]))

    def _load_point_cloud(self, file_path):
        # file_path = os.path.join(self.data_path, file_path)
        
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
        if self.mix_finetune:
            return len(self.anno_pretrain)
        else:
            return len(self.anno)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        """
        r = np.random.rand()
        if self.mix_finetune and r < self.finetune_ratio:
            data = np.random.choice(self.anno)
            file_path = os.path.join(self.dataset_path, 'rollout', data['file_path'])
        elif self.mix_finetune:
            data = self.anno_pretrain[index]
            file_path = os.path.join(self.pretrain_path, 'rollout', data['file_path'])
        else:
            data = self.anno[index]
            file_path = os.path.join(self.dataset_path, 'rollout', data['file_path'])

        # Load point cloud and target point
        point_cloud = self._load_point_cloud(file_path)
        target_point = np.array(data['pose']['se2'], dtype=np.float32)
        label = 1 if data['is_success'] else 0
        if self.settings is not None:
            task_idx = self.settings.index(data['task_name'])
        else:
            task_idx = 0

        # Normalize both point cloud and target point together
        # point_cloud, target_point = self.pc_norm(point_cloud, target_point)

        # Ensure point cloud has consistent size
        point_cloud = fix_point_cloud_size(point_cloud, self.pointnum)

        # Apply augmentations
        if 'augmentations' in self.config:
            point_cloud, target_point = apply_augmentations(point_cloud, target_point, self.config['augmentations'])

        # Convert to torch tensors
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        target_point = torch.from_numpy(target_point)
        label = torch.tensor(label, dtype=torch.long)
        task_idx = torch.tensor(task_idx, dtype=torch.long)
        
        return {
            'point_cloud': point_cloud,
            'target_point': target_point,
            'label': label,
            'task_idx': task_idx
        }
