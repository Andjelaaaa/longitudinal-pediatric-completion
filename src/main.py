import argparse
import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from training import train_model
from loader import CP
from model import DenoisingNetwork, DenoisingNetworkParallel
from accelerate import Accelerator

def main(use_accelerator, use_data_parallel):
    # Initialize accelerator if model parallelism is requested
    accelerator = Accelerator(mixed_precision="fp16") if use_accelerator else None

    # Set device and load data
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    romane_dir = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    train_dataset = CP(root_dir=romane_dir, age_csv=f'{romane_dir}/trios_sorted_by_age.csv', transfo_type='rigid')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Select model
    model = DenoisingNetworkParallel(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128) \
            if use_accelerator else DenoisingNetwork(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)

    # Use DataParallel if specified and not using `Accelerate`
    if use_data_parallel and not use_accelerator:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])

    # Prepare model and data for distributed training if accelerator is used
    if use_accelerator:
        model, train_loader = accelerator.prepare(model, train_loader)
    else:
        model.to(device)

    # Count the total number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Verify that all parameters are on the correct devices
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

    # Define noise schedule
    noise_schedule = torch.linspace(1e-4, 5e-3, 1000, dtype=torch.float32).to(device)

    # Train the model
    train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, accelerator=accelerator)

    # Save the model only on the main process
    if not use_accelerator or accelerator.is_main_process:
        torch.save(model.state_dict(), "checkpoints/model.pth")

    # Shutdown the process group to avoid NCCL warnings
    if torch.distributed.is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train the model with data or model parallelism.")
    parser.add_argument("--use_accelerator", type=bool, default=False, 
                        help="Use accelerator for model parallelism.")
    parser.add_argument("--use_data_parallel", type=bool, default=False, 
                        help="Use DataParallel for data parallelism.")

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(use_accelerator=args.use_accelerator, use_data_parallel=args.use_data_parallel)









# Main training function
# def main():
#     # Initialize wandb
#     wandb.init(project="long-ped-comp", entity="adimitri")

#     # TensorBoard setup
#     writer = SummaryWriter(log_dir="./logs")

#     # Data loading and preprocessing
#     train_dataset = CP(root_dir='/CP/sub-001', age_csv='/path/to/trios_sorted_by_age.csv')
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#     # Model building
#     input_shape = (64, 64, 64, 1)  # Example input shape for the model
#     model = DenoisingNetwork(input_shape=input_shape, filters=64, age_embedding_dim=128)

#     # Noise schedule for the diffusion process
#     noise_schedule = torch.linspace(1e-4, 5e-3, 1000).to('cuda' if torch.cuda.is_available() else 'cpu')

#     # Move model to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Optimizer
#     optimizer = Adam(model.parameters(), lr=2e-4)

#     # Training parameters
#     epochs = 10
#     lambda_fusion = 0.6

#     # Training the model
#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}/{epochs}")
#         for step, inputs in enumerate(train_loader):
#             # Perform a single training step
#             loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device)

#             # Log the loss
#             if step % 10 == 0:
#                 print(f"Step {step}, Loss: {loss.item()}")
#                 wandb.log({"loss": loss.item()})
#                 writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + step)

#         # Log the losses to wandb at the end of each epoch
#         wandb.log({"epoch": epoch + 1, "loss": loss.item()})

#     # Optionally, save the model
#     torch.save(model.state_dict(), "model.pth")

#     # Close the TensorBoard writer
#     writer.close()


