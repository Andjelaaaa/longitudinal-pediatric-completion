import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from models.build_model import build_model_3d
from utils.training import train_model, train_step
from loader import load_and_preprocess_data

def main():
    # Initialize wandb
    wandb.init(project="long-ped-comp", entity="adimitri")

    # TensorBoard setup
    writer = SummaryWriter(log_dir="./logs")

    # Data loading and preprocessing
    train_dataset = load_and_preprocess_data()  # Implement this function to load your data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Model building
    input_shape = (64, 64, 64, 1)  # Example input shape
    model, noise_schedule = build_model_3d(input_shape)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Training parameters
    epochs = 10
    lambda_fusion = 0.6

    # Training the model
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, inputs in enumerate(train_loader):
            inputs = [x.to(device) for x in inputs]
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device)

            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + step)

        # Log the losses to wandb at the end of each epoch
        wandb.log({"epoch": epoch + 1, "loss": loss.item()})

    # Optionally, save the model
    torch.save(model.state_dict(), "model.pth")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
