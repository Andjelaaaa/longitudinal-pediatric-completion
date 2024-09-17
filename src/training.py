import torch
import wandb

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

# Training step function
def train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device):
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

    # Move inputs to the correct device (GPU or CPU)
    p, t, s, age = p.to(device), t.to(device), s.to(device), age.to(device)
    
    # Generate a random noise level based on the noise schedule
    noise_level = torch.rand(1).item() * (noise_schedule[-1] - noise_schedule[0]) + noise_schedule[0]
    
    # Add noise to the target image to simulate noisy observations
    noisy_t = t + noise_level * torch.randn_like(t).to(device)
    
    # Ground truth noise (eps) to be removed from the noisy target image
    eps = torch.randn_like(t).to(device)  # Ground truth noise

    # Zero out gradients before backward pass
    optimizer.zero_grad()

    # Forward pass through the model
    predicted_eps, loci_outputs_p, loci_outputs_s = model(p, noisy_t, s, age)

    # Assuming predicted_eps is the model's prediction for the noise to remove from noisy_t
    # Cfused should be the main prediction output, here it's just predicted_eps
    c_fused = predicted_eps  # In this case, c_fused corresponds to predicted noise correction
    predicted_eps = c_fused - noisy_t  # The predicted noise to be subtracted

    # Calculate the total loss
    loss = total_loss(eps, predicted_eps, loci_outputs_p, loci_outputs_s, lambda_fusion)
    
    # Backward pass (compute gradients)
    loss.backward()
    
    # Update model weights
    optimizer.step()

    return loss

# Main training function
def train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, device="cpu"):
    """
    Main training loop:
    1. Iterate through the dataset for a given number of epochs
    2. Perform a forward and backward pass for each batch
    3. Log training progress
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # Optimizer for updating model weights

    # Initialize WandB for tracking metrics
    wandb.init(project="long-ped-comp", entity="adimitri")
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for step, inputs in enumerate(train_loader):
            # Move inputs to the device and perform a training step
            inputs = [x.to(device) for x in inputs]
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device)

            # Accumulate the loss for epoch logging
            epoch_loss += loss.item()

            # Log step-wise loss
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

        # Log the average loss for the epoch
        epoch_loss /= len(train_loader)
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    # Optionally, save the model at the end of training
    torch.save(model.state_dict(), "model.pth")
