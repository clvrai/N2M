import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse
import wandb
import time
import datetime
import random
import numpy as np

from n2m.data.dataset import make_data_module
from n2m.model.N2Mnet import N2Mnet
from n2m.utils.config import *
from n2m.utils.visualizer import save_gmm_visualization, save_gmm_visualization_se2, save_gmm_visualization_xythetaz
from n2m.utils.loss import Loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device, train_config):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Move data to device
        point_cloud = batch['point_cloud'].to(device)
        target_point = batch['target_point'].to(device)
        label = batch['label'].to(device)
        
        # Forward pass
        means, covs, weights = model(point_cloud)
        loss = loss_fn(means, covs, weights, target_point, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
            
    
    avg_loss = total_loss / len(train_loader)
    wandb.log({"train/loss": avg_loss, "epoch": epoch})
    return avg_loss

def validate(model, val_loader, loss_fn, epoch, device, val_dir,train_config):
    model.eval()
    total_loss = 0
    
    # Create visualization directory
    os.makedirs(val_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
            # Move data to device
            point_cloud = batch['point_cloud'].to(device)
            target_point = batch['target_point'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass
            means, covs, weights = model(point_cloud)
            loss = loss_fn(means, covs, weights, target_point, label)
            
            # Update metrics
            total_loss += loss.item()
            
            # Generate visualization for first item in batch
            for i in range(len(batch['point_cloud'])):
                if target_point[i].shape[0] == 3:
                    save_gmm_visualization_se2(
                        point_cloud[i].cpu().numpy(),
                        target_point[i].cpu().numpy(),
                        label[i].cpu().numpy(),
                        means[i].cpu().numpy(),
                        covs[i].cpu().numpy(),
                        weights[i].cpu().numpy(),
                        os.path.join(val_dir, f'batch_{batch_idx}_{i}.pcd')
                    )
                elif target_point[i].shape[0] == 4:
                    save_gmm_visualization_xythetaz(
                        point_cloud[i].cpu().numpy(),
                        target_point[i].cpu().numpy(),
                        label[i].cpu().numpy(),
                        means[i].cpu().numpy(),
                        covs[i].cpu().numpy(),
                        weights[i].cpu().numpy(),
                        os.path.join(val_dir, f'batch_{batch_idx}_{i}.pcd')
                    )
        
    
    avg_loss = total_loss / len(val_loader)
    
    # Log to wandb
    wandb.log({"val/loss": avg_loss, "epoch": epoch})
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def get_exp_dir(train_config, use_time_str):
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
    
    # Determine base output directory
    if train_config.get('output_dir') is not None:
        base_output_dir = train_config['output_dir']
    else:
        # Use dataset_path/training as default
        dataset_path = train_config['dataset']['dataset_path']
        base_output_dir = os.path.join(dataset_path, 'training')
    
    if use_time_str:
        output_dir = os.path.join(base_output_dir, time_str)
    else:
        output_dir = base_output_dir    

    ckpt_dir = os.path.join(output_dir, 'ckpts')
    val_dir = os.path.join(output_dir, 'val')
    log_dir = os.path.join(output_dir, 'logs')

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return output_dir, ckpt_dir, val_dir, log_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to the config file.')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the dataset directory.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--use_time_str', type=bool, default=False, help='Use time string as output directory name.')
    args = parser.parse_args()
    
    # Load configuration from the same directory as this script
    if args.config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', 'configs', 'config.json')
    else:
        config_path = args.config
    
    config = load_config(config_path)
    train_config = config['train']
    n2mnet_config = config['n2mnet']
    dataset_config = train_config['dataset']
    
    # Override with command line arguments
    if args.dataset_path is not None:
        dataset_config['dataset_path'] = args.dataset_path  
    
    if args.encoder_ckpt is not None:
        train_config['encoder']['ckpt'] = args.encoder_ckpt

    # Create output directory
    output_dir, ckpt_dir, val_dir, log_dir = get_exp_dir(train_config, args.use_time_str)

    # Initialize wandb
    wandb_config = train_config['wandb']
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        name=wandb_config['name'],
        config=config,
        mode="online" if wandb_config['entity'] is not None else "disabled"
    )

    # save config
    config_save_path = os.path.join(output_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Merge encoder config with n2mnet config for model initialization
    model_config = {
        'encoder': {
            **n2mnet_config['encoder'],
            'ckpt': train_config['encoder']['ckpt'],
            'freeze': train_config['encoder']['freeze']
        },
        'decoder': n2mnet_config['decoder']
    }
    
    # Create model
    model = N2Mnet(
        config=model_config
    ).to(device)

    # Watch model with wandb
    wandb.watch(model, log="all", log_freq=100)
    
    # Create data loaders
    train_dataset, val_dataset = make_data_module(dataset_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=True,
        prefetch_factor=4
    )
    if len(val_loader) == 0:
        val_loader = train_loader
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config['learning_rate']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=int(train_config['num_epochs'])
    )
    
    # create loss function
    loss_config = train_config['loss']
    loss_fn = Loss(loss_config)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(train_config['num_epochs']):
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device, train_config)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
        
        # Validate
        if (epoch + 1) % train_config['val_freq'] == 0:
            val_loss = validate(model, val_loader, loss_fn, epoch, device, val_dir, train_config)
            print(f'Epoch {epoch}: Val Loss = {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    os.path.join(ckpt_dir, 'best_model.pth')
                )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                os.path.join(ckpt_dir, f'model_{epoch}.pth')
            )
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        train_config['num_epochs'],
        train_loss,
        os.path.join(ckpt_dir, 'final_model.pth')
    )
    
    wandb.finish()

if __name__ == '__main__':
    main()
