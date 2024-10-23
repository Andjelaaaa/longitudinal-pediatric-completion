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
    # Diffusion loss (for predicting the added noise)
    l_diff = diffusion_loss(eps, predicted_eps)
    
    # Fusion loss (consistency between preceding and subsequent image features)
    l_fusion = fusion_loss(c_pred_p, c_pred_s)
    
    return l_diff + lambda_fusion * l_fusion


# Training step function with Accelerator support
def train_step(model, optimizer, inputs, accelerator=None, lambda_fusion=0.6):
    """
    Perform a single training step:
    1. Forward pass
    2. Compute loss
    3. Backward pass
    4. Update model weights
    """
    # Determine the device
    device = accelerator.device if accelerator else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Input tensor is on device: {p.device}")

    # Move inputs to the selected device and ensure they are float32
    p, t, s, age = [x.to(device).float() for x in inputs]

    # Zero out gradients
    optimizer.zero_grad()

    # Forward pass with autocast for mixed precision (only if accelerator is used)
    if accelerator:
        with accelerator.autocast():
            # Model forward pass (loss is computed inside the model)
            loss = model(p, t, s, age, lambda_fusion)
        accelerator.backward(loss)
    else:
        # Standard forward pass
        loss = model(p, t, s, age, lambda_fusion)
        loss.backward()

    optimizer.step()

    return loss

def monitor_data_batch(inputs):
    total_data_size = sum([input.element_size() * input.nelement() for input in inputs]) / 1024**2
    print(f"Batch size: {total_data_size:.2f} MB")
    for i, input_tensor in enumerate(inputs):
        print(f"Input {i} | Size: {input_tensor.size()} | Memory: {input_tensor.element_size() * input_tensor.nelement() / 1024**2:.2f} MB")

# Main training function
def train_model(model, train_loader, epochs=10, lambda_fusion=0.6, accelerator=None):
    """
    Main training loop:
    1. Iterate through the dataset for a given number of epochs
    2. Perform a forward and backward pass for each batch
    3. Log training progress
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # If an accelerator is provided, use it to prepare model, optimizer, and dataloader
    if accelerator:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Initialize WandB for tracking metrics (only on the main process)
    if accelerator is None or accelerator.is_main_process:
        wandb.init(project="long-ped-comp", entity="adimitri")
        wandb.watch(model, log="all")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for step, inputs in enumerate(train_loader):
            monitor_data_batch(inputs)
            
            # Perform a training step
            loss = train_step(model, optimizer, inputs, accelerator=accelerator, lambda_fusion=lambda_fusion)

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Log step-wise loss (only on main process)
            if step % 10 == 0 and (accelerator is None or accelerator.is_main_process):
                print(f"Step {step}, Loss: {loss.item()}")
                wandb.log({"batch_loss": loss.item(), "epoch": epoch + 1})

        # Log epoch loss
        epoch_loss /= len(train_loader)
        if accelerator is None or accelerator.is_main_process:
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    # Save the model state (only on main process)
    if accelerator is None or accelerator.is_main_process:
        torch.save(model.state_dict(), "model.pth")

    # Wait for all processes to finish
    if accelerator:
        accelerator.wait_for_everyone()
