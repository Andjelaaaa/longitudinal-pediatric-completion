import torch
import torch.nn as nn
from utils.losses import total_loss

def train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device):
    model.train()
    
    p, t, s = inputs
    noise_level = torch.rand(1).item() * (noise_schedule[-1] - noise_schedule[0]) + noise_schedule[0]
    noisy_t = t + noise_level * torch.randn_like(t).to(device)
    eps = torch.randn_like(t).to(device)  # Ground truth noise
    
    optimizer.zero_grad()

    predictions, loci_outputs_p, loci_outputs_s = model(p, noisy_t, s)
    c_fused = predictions  # Assuming c_fused is the main prediction output
    predicted_eps = c_fused - noisy_t  # Predicted noise
    
    loss = total_loss(eps, predicted_eps, loci_outputs_p, loci_outputs_s, lambda_fusion)
    
    loss.backward()
    optimizer.step()
    
    return loss

def train_model(model, train_loader, noise_schedule, epochs=10, lambda_fusion=0.6, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, inputs in enumerate(train_loader):
            inputs = [x.to(device) for x in inputs]
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion, device)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
