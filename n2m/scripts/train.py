import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse
import wandb

from nav2man.data.SIRDataset import make_SIR_data_module
from nav2man.model.SIRPredictor import SIRPredictor
from nav2man.utils.config import *
from nav2man.utils.visualizer import save_gmm_visualization, save_gmm_visualization_se2, save_gmm_visualization_xythetaz
from nav2man.utils.loss import Loss
import random
import numpy as np

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
    
    if 'PTR' not in train_config:
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
    else:
        for batch in pbar:
            # Move data to device
            point_cloud = batch['point_cloud'].to(device)
            target_point = batch['target_point'].to(device)
            label = batch['label'].to(device)
            task_idx = batch['task_idx'].to(device)
            
            # Forward pass
            means, covs, weights = model(point_cloud, task_idx)
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

def validate(model, val_loader, loss_fn, epoch, device, train_config):
    model.eval()
    total_loss = 0
    
    # Create visualization directory
    vis_dir = os.path.join(train_config['output_dir'], 'visualizations', f'val')
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        if 'PTR' not in train_config:
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
                            os.path.join(vis_dir, f'batch_{batch_idx}_{i}.ply')
                        )
                    elif target_point[i].shape[0] == 4:
                        save_gmm_visualization_xythetaz(
                            point_cloud[i].cpu().numpy(),
                            target_point[i].cpu().numpy(),
                            label[i].cpu().numpy(),
                            means[i].cpu().numpy(),
                            covs[i].cpu().numpy(),
                            weights[i].cpu().numpy(),
                            os.path.join(vis_dir, f'batch_{batch_idx}_{i}.ply')
                        )
        else:
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                # Move data to device
                point_cloud = batch['point_cloud'].to(device)
                target_point = batch['target_point'].to(device)
                label = batch['label'].to(device)
                task_idx = batch['task_idx'].to(device)
                
                # Forward pass
                means, covs, weights = model(point_cloud, task_idx)
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
                            os.path.join(vis_dir, f'batch_{batch_idx}_{i}.ply')
                        )
                    elif target_point[i].shape[0] == 4:
                        save_gmm_visualization_xythetaz(
                            point_cloud[i].cpu().numpy(),
                            target_point[i].cpu().numpy(),
                            label[i].cpu().numpy(),
                            means[i].cpu().numpy(),
                            covs[i].cpu().numpy(),
                            weights[i].cpu().numpy(),
                            os.path.join(vis_dir, f'batch_{batch_idx}_{i}.ply')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/test.yml')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    train_config = config['train']
    model_config = config['model']
    dataset_config = config['dataset']

    # Create output directory
    os.makedirs(train_config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(train_config['output_dir'], 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(train_config['output_dir'], 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(train_config['output_dir'], 'logs'), exist_ok=True)

    # Initialize wandb
    wandb_config = train_config['wandb']
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        name=wandb_config['name'],
        config=config
    )

    # save config
    config_save_path = os.path.join(train_config['output_dir'], 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    model = SIRPredictor(
        config=model_config
    ).to(device)

    # Watch model with wandb
    wandb.watch(model, log="all", log_freq=100)
    
    # Create data loaders
    PTR = train_config['PTR'] if 'PTR' in train_config else None
    train_dataset, val_dataset = make_SIR_data_module(dataset_config, PTR)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config['num_workers'],
        pin_memory=True
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
            val_loss = validate(model, val_loader, loss_fn, epoch, device, train_config)
            print(f'Epoch {epoch}: Val Loss = {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    os.path.join(train_config['output_dir'], 'ckpts', 'best_model.pth')
                )
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                os.path.join(train_config['output_dir'], 'ckpts', f'model_{epoch}.pth')
            )
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        train_config['num_epochs'],
        train_loss,
        os.path.join(train_config['output_dir'], 'ckpts', 'final_model.pth')
    )
    
    wandb.finish()

if __name__ == '__main__':
    main()
