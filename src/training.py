import torch
import wandb
from accelerate import Accelerator

# Loss functions
def diffusion_loss(eps, predicted_eps):
    return torch.mean((eps - predicted_eps) ** 2)

def fusion_loss(c_pred_p, c_pred_s):
    return torch.mean((c_pred_p - c_pred_s) ** 2)

def total_loss(eps, predicted_eps, c_pred_p, c_pred_s, lambda_fusion=0.6):
    """
    Compute the total loss by combining the diffusion loss and fusion loss.
    """
    l_diff = diffusion_loss(eps, predicted_eps)
    l_fusion = fusion_loss(c_pred_p, c_pred_s)
    return l_diff + lambda_fusion * l_fusion

# Training step function with Accelerator support
def train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, accelerator):
    """
    Perform a single training step:
    1. Forward pass
    2. Compute loss
    3. Backward pass
    4. Update model weights
    """
    model.train()  # Ensure the model is in training mode

    # Unpack inputs: p (preceding), t (target), s (subsequent), age (age tensor)
    p, t, s, age = inputs

    # Generate a random noise level based on the noise schedule
    noise_level = torch.rand(1).item() * (noise_schedule[-1] - noise_schedule[0]) + noise_schedule[0]
    noisy_t = t + noise_level * torch.randn_like(t).to(t.device)

    # Ground truth noise (eps) to be removed from the noisy target image
    eps = torch.randn_like(t).to(t.device)

    # Zero out gradients before backward pass
    optimizer.zero_grad()

    # Forward pass through the model
    with accelerator.autocast():
        predicted_eps, loci_outputs_p, loci_outputs_s = model(p, noisy_t, s, age)

        # Adjust predicted_eps to align with the noise schedule
        c_fused = predicted_eps
        predicted_eps = c_fused - noisy_t  # Predicted noise to be subtracted

        # Calculate the total loss
        loss = total_loss(eps, predicted_eps, loci_outputs_p, loci_outputs_s, lambda_fusion)

    # Backward pass (compute gradients) and optimizer step with accelerator support
    accelerator.backward(loss)
    optimizer.step()

    return loss

# Main training function
def train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, accelerator=None):
    """
    Main training loop:
    1. Iterate through the dataset for a given number of epochs
    2. Perform a forward and backward pass for each batch
    3. Log training progress
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Prepare model, optimizer, and data loader for distributed training
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Initialize WandB for tracking metrics (only on the main process)
    if accelerator.is_main_process:
        wandb.init(project="long-ped-comp", entity="adimitri")
        wandb.watch(model, log="all")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for step, inputs in enumerate(train_loader):
            # Perform a training step with the accelerator
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, accelerator)

            # Accumulate the loss for epoch logging
            epoch_loss += loss.item()

            # Log step-wise loss (only on the main process)
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Step {step}, Loss: {loss.item()}")
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

        # Log the average loss for the epoch (only on the main process)
        epoch_loss /= len(train_loader)
        if accelerator.is_main_process:
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    # Save the model state (only on the main process)
    if accelerator.is_main_process:
        torch.save(model.state_dict(), "model.pth")

    # Wait for all processes to finish (barrier)
    accelerator.wait_for_everyone()
