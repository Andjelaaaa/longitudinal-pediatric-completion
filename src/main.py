import torch
from torch.utils.data import DataLoader
from training import train_model
from loader import CP  # Custom dataset class
from model import DenoisingNetwork, DenoisingNetworkParallel  # Import both versions

def main(use_accelerator=False):
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CP dataset
    romane_dir = '/home/GRAMES.POLYMTL.CA/andim/joplin-intra-inter/CP_rigid_trios/CP'
    train_dataset = CP(root_dir=romane_dir, age_csv=f'{romane_dir}/trios_sorted_by_age.csv', transfo_type='rigid')

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Choose the appropriate model
    if use_accelerator:
        print("Using parallel DenoisingNetwork with multiple GPUs")
        model = DenoisingNetworkParallel(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)
    else:
        print("Using standard DenoisingNetwork")
        model = DenoisingNetwork(input_shape=(1, 256, 256, 105), filters=64, age_embedding_dim=128)

    # Move model to the appropriate device
    model.to(device)

    # Define the noise schedule with float32 dtype
    noise_schedule = torch.linspace(1e-4, 5e-3, 1000, dtype=torch.float32).to(device)

    # Train the model
    train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6)

    # Save the trained model
    torch.save(model.state_dict(), "checkpoints/model.pth")

if __name__ == "__main__":
    # Run without accelerator by default
    main(use_accelerator=False)






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


