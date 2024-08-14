import torch
import torch.nn.functional as F

def diffusion_loss(eps, predicted_eps):
    return torch.mean((eps - predicted_eps) ** 2)

def fusion_loss(c_pred_p, c_pred_s):
    return torch.mean((c_pred_p - c_pred_s) ** 2)

def total_loss(eps, predicted_eps, c_pred_p, c_pred_s, lambda_fusion=0.6):
    l_diff = diffusion_loss(eps, predicted_eps)
    l_fusion = fusion_loss(c_pred_p, c_pred_s)
    return l_diff + lambda_fusion * l_fusion
