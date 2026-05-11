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
    
    # Enable mixed precision training if available and not disabled
    use_amp = train_config.get('use_amp', True) and torch.cuda.is_available() and not train_config.get('disable_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) training")
    
    for batch in pbar:
        # Move data to device with non_blocking for faster transfer
        point_cloud = batch['point_cloud'].to(device, non_blocking=True)
        target_point = batch['target_point'].to(device, non_blocking=True)
        label = batch['label'].to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                means, covs, weights = model(point_cloud)
                loss = loss_fn(means, covs, weights, target_point, label)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward pass
            means, covs, weights = model(point_cloud)
            loss = loss_fn(means, covs, weights, target_point, label)
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

def save_checkpoint(model, optimizer, epoch, loss, save_path, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint and return epoch and loss"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Resumed from epoch {epoch} with loss {loss:.4f}")
    return epoch, loss

def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint (highest epoch) in the checkpoint directory"""
    if not os.path.exists(ckpt_dir):
        return None
    
    # Look for model_*.pth files
    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith('model_') and f.endswith('.pth')]
    
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and get the latest (highest epoch)
    try:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        latest_checkpoint = os.path.join(ckpt_dir, checkpoint_files[-1])
        epoch_num = int(checkpoint_files[-1].split('_')[1].split('.')[0])
        print(f"Found latest checkpoint: {checkpoint_files[-1]} (epoch {epoch_num})")
        return latest_checkpoint
    except (ValueError, IndexError) as e:
        print(f"Error parsing checkpoint filenames: {e}")
        return None

def get_best_val_loss(ckpt_dir):
    """Get the best validation loss from best_model.pth if it exists"""
    best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"Found best validation loss: {best_loss:.4f}")
            return best_loss
        except Exception as e:
            print(f"Error reading best model: {e}")
    return float('inf')

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
    parser.add_argument('--use_cache', action='store_true', help='Cache all training data in memory for faster training.')
    parser.add_argument('--cache_threads', type=int, default=64, help='Number of threads for data caching (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=32, help='Override batch size from config')
    parser.add_argument('--max_epoch', type=int, default=300, help='Maximum number of epochs to train')
    parser.add_argument('--num_gaussians', type=int, default=1, help='Number of Gaussian components for GMM')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension: 3 (xytheta) or 4 (xyztheta)')
    parser.add_argument('--no_resume', action='store_true', help='Disable auto-resume and start training from scratch')
    parser.add_argument('--disable_amp', action='store_true', help='Disable automatic mixed precision training')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--no_val', action='store_true', help='Skip validation and save checkpoint directly at save intervals')
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
    
    # Override config with command line arguments
    if hasattr(args, 'output_dim'):
        n2mnet_config['decoder']['output_dim'] = args.output_dim
    
    # Override with command line arguments
    if args.dataset_path is not None:
        dataset_config['dataset_path'] = args.dataset_path  
    
    if args.encoder_ckpt is not None:
        train_config['encoder']['ckpt'] = args.encoder_ckpt
    
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
        print(f"Using custom batch size: {args.batch_size}")
    
    # Override max epochs (always apply since it has a default value)
    train_config['num_epochs'] = args.max_epoch
    print(f"Using max epochs: {args.max_epoch}")
    
    # Override num_gaussians in decoder config
    config['n2mnet']['decoder']['num_gaussians'] = args.num_gaussians
    print(f"Using num_gaussians: {args.num_gaussians}")
    
    # Add use_cache parameter to dataset config
    dataset_config['use_cache'] = args.use_cache
    if args.cache_threads is not None:
        dataset_config['cache_threads'] = args.cache_threads
    
    # Add performance optimization flags to train config
    train_config['disable_amp'] = args.disable_amp
    train_config['profile'] = args.profile
    
    if args.use_cache:
        print("WARNING: Data caching is enabled. This will load all training data into memory.")
        print("Make sure you have sufficient RAM available.")
        if args.cache_threads is not None:
            print(f"Using {args.cache_threads} threads for data loading.")
    
    if args.profile:
        print("Performance profiling enabled")

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
    
    # Optimize DataLoader settings based on caching
    if args.use_cache:
        # When using cache, minimal workers since data is already in memory
        num_workers = 1  # No workers needed for cached data
        prefetch_factor = 1
        print(f"Using optimized DataLoader settings for cached data: {num_workers} workers")
    else:
        num_workers = train_config['num_workers']
        prefetch_factor = 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0
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
    
    # Create training log file
    log_file_path = os.path.join(log_dir, 'training_log.txt')
    
    # Initialize log file with header if starting fresh, or append if resuming
    def write_log_header():
        with open(log_file_path, 'w') as f:
            f.write("N2M Training Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Dataset: {dataset_config['dataset_path']}\n")
            f.write(f"Batch Size: {train_config['batch_size']}\n")
            f.write(f"Max Epochs: {train_config['num_epochs']}\n")
            f.write(f"Learning Rate: {train_config['learning_rate']}\n")
            f.write(f"Validation Frequency: Every {train_config['val_freq']} epochs\n")
            f.write("=" * 50 + "\n")
            f.write("Epoch\tTrain_Loss\tVal_Loss\tBest_Val_Loss\tTimestamp\n")
    
    def append_log(epoch, train_loss, val_loss=None, best_val_loss=None):
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file_path, 'a') as f:
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            best_val_loss_str = f"{best_val_loss:.4f}" if best_val_loss is not None else "N/A"
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss_str}\t{best_val_loss_str}\t{timestamp}\n")
    
    # Auto-resume training if checkpoint exists (unless disabled)
    start_epoch = 0
    best_val_loss = float('inf')
    
    if not args.no_resume:
        checkpoint_path = find_latest_checkpoint(ckpt_dir)
        if checkpoint_path:
            try:
                resume_epoch, resume_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
                start_epoch = resume_epoch + 1  # Start from next epoch
                # Get the best validation loss from best_model.pth
                best_val_loss = get_best_val_loss(ckpt_dir)
                print(f"✅ Auto-resumed training from epoch {start_epoch}")
                # Don't overwrite log file when resuming
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                print("Starting training from scratch...")
                write_log_header()
        else:
            print("No checkpoint found, starting training from scratch...")
            write_log_header()
    else:
        print("Auto-resume disabled, starting training from scratch...")
        write_log_header()
    
    # Training loop
    for epoch in range(start_epoch, train_config['num_epochs']):
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device, train_config)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
        
        # Validate or save checkpoint directly
        if (epoch + 1) % train_config['val_freq'] == 0:
            if args.no_val:
                # Skip validation, save checkpoint directly
                print(f'Epoch {epoch}: Skipping validation, saving checkpoint directly')
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    os.path.join(ckpt_dir, f'model_{epoch}.pth'),
                    scheduler
                )
                # Log training progress without validation
                append_log(epoch, train_loss)
            else:
                # Normal validation process
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
                        os.path.join(ckpt_dir, 'best_model.pth'),
                        scheduler
                    )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    train_loss,
                    os.path.join(ckpt_dir, f'model_{epoch}.pth'),
                    scheduler
                )
                
                # Log training progress with validation (only when validation occurs)
                append_log(epoch, train_loss, val_loss, best_val_loss)
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    save_checkpoint(
        model,
        optimizer,
        train_config['num_epochs'],
        train_loss,
        os.path.join(ckpt_dir, 'final_model.pth'),
        scheduler
    )
    
    wandb.finish()

if __name__ == '__main__':
    main()
