import argparse
import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from training import train_model  
from loader import CP  # Custom dataset loader
from model import DPM, GAMUNet, FusionModule  # Import your model classes
from accelerate import Accelerator
import numpy as np
import tqdm

def train():
    device = torch.device("cuda")

    # DPM hyperparameters
    n_T = 1000  # Number of diffusion timesteps

    # Set the noise schedule from 1e-4 to 5e-3 linearly over n_T steps
    betas = (1e-4, 5e-3)

    # andjela_dir = '/home/andjela/joplin-intra-inter/CP_rigid_trios/CP'
    # train_dataset = CP(root_dir=andjela_dir, age_csv=f'{andjela_dir}/trios_sorted_by_age.csv', transfo_type='rigid')
    romane_dir = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    train_dataset = CP(root_dir=romane_dir, age_csv=f'{romane_dir}/trios_sorted_by_age.csv', transfo_type='rigid')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=5)

    # valid_loader = DataLoader(ACDCDataset(data_dir="data", split="val"), batch_size=batch_size, shuffle=False, num_workers=1)
    # x_val, x_prev_val = next(iter(valid_loader))
    # x_prev_val = x_prev_val.to(device)


    fusion_model = FusionModule(in_channels=1, filters=64, age_embedding_dim=128, num_repeats=4)
    gam_unet = GAMUNet(in_channels=1, age_embedding_dim=128)

    dpm = DPM(fusion_model=fusion_model, nn_model=gam_unet, betas=betas, n_T=n_T, device=device)
    dpm.to(device)

    # Call the train_model function to start the training process
    train_model(dpm, train_loader, epochs=10, lambda_fusion=0.6)

    
# # Register hooks to track memory usage
# def memory_hook(module, input, output):
#     print(f"{module.__class__.__name__} | Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")

# def register_hooks(model):
#     for layer in model.modules():
#         if isinstance(layer, torch.nn.Module):
#             layer.register_forward_hook(memory_hook)

# def main(use_accelerator, use_data_parallel):
#     # Initialize accelerator if used
#     accelerator = Accelerator(mixed_precision="fp16") if use_accelerator else None

#     # Set device and load data
#     device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Path to dataset (adjust to your environment)
#     # romane_dir = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
#     andjela_dir = '/home/andjela/joplin-intra-inter/CP_rigid_trios/CP'
#     train_dataset = CP(root_dir=andjela_dir, age_csv=f'{andjela_dir}/trios_sorted_by_age.csv', transfo_type='rigid')
#     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=4)

#     # Monitor memory before data loading
#     print(f"Memory allocated before loading data: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     inputs = next(iter(train_loader))  # Load a batch of data
#     print(f"Memory allocated after loading data: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

#     # Select the model based on whether accelerator is used
#     if use_accelerator:
#         model = DenoisingNetworkParallel(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)
#     else:
#         model = DenoisingNetwork(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)

#     # Register memory hooks to monitor usage
#     register_hooks(model)

#     # Prepare model and data for distributed training if accelerator is used
#     if use_accelerator:
#         model, train_loader = accelerator.prepare(model, train_loader)
#     elif use_data_parallel:
#         model = torch.nn.DataParallel(model, device_ids=[0, 1])

#     # Move the model to the selected device
#     print(f"Using device: {device}")
#     model.to(device)

#     # Check memory after model creation
#     print(f"Memory allocated after model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

#     # Noise schedule for training
#     noise_schedule = torch.linspace(1e-4, 5e-3, 1000, dtype=torch.float32).to(device)

#     # Call the train_model function to start the training process
#     train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, accelerator=accelerator)

#     # Save the model state (only on the main process if using distributed training)
#     if not use_accelerator or accelerator.is_main_process:
#         torch.save(model.state_dict(), "checkpoints/model.pth")

#     # Shutdown process group to avoid NCCL warnings
#     if torch.distributed.is_initialized():
#         destroy_process_group()

if __name__ == "__main__":
    # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Train the model with or without an accelerator.")
    # parser.add_argument("--use_accelerator", type=bool, default=False, help="Use accelerator for FSDP.")
    # parser.add_argument("--use_data_parallel", type=bool, default=False, help="Use DataParallel for multi-GPU training.")
    # args = parser.parse_args()

    # # Run the main function with the parsed arguments
    # main(use_accelerator=args.use_accelerator, use_data_parallel=args.use_data_parallel)
    train()

