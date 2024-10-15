import torch
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from training import train_model
from loader import CP
from model import DenoisingNetwork, DenoisingNetworkParallel
from accelerate import Accelerator

def main(use_accelerator=False):
    # Initialize accelerator if used
    accelerator = Accelerator(mixed_precision="fp16") if use_accelerator else None

    # Set device and load data
    device = accelerator.device if accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    romane_dir = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    train_dataset = CP(root_dir=romane_dir, age_csv=f'{romane_dir}/trios_sorted_by_age.csv', transfo_type='rigid')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Select model
    model = DenoisingNetworkParallel(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128) \
            if use_accelerator else DenoisingNetwork(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)

    # Prepare model and data for distributed training if accelerator is used
    if use_accelerator:
        model, train_loader = accelerator.prepare(model, train_loader)
    else:
        model.to(device)

    # Verify that all parameters are on the correct devices
    for name, param in model.named_parameters():
        print(f"{name} is on {param.device}")

    # Noise schedule
    noise_schedule = torch.linspace(1e-4, 5e-3, 1000, dtype=torch.float32).to(device)

    # Train model
    train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, accelerator=accelerator)

    # Save model on main process only
    if not use_accelerator or accelerator.is_main_process:
        torch.save(model.state_dict(), "checkpoints/model.pth")

    # Shutdown process group to avoid NCCL warnings
    if torch.distributed.is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    main(use_accelerator=True)







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


