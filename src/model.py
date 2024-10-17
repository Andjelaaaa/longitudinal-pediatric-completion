import torch
import torch.nn as nn
import torch.nn.functional as F

# Age Embedding
class AgeEmbedding(nn.Module):
    """
    Embeds the age input as textual features.
    """
    def __init__(self, embedding_dim=128):
        super(AgeEmbedding, self).__init__()
        self.fc = nn.Linear(1, embedding_dim)  # From scalar age to embedding_dim
        self.relu = nn.ReLU()

    def forward(self, age):
        # age is expected to be a scalar (1D tensor) or [batch_size, 1]
        return self.relu(self.fc(age))

# Cross-Attention Module
class CrossAttention(nn.Module):
    """
    Cross-attention between pixel features and age features or between different image features.
    """
    def __init__(self, pixel_dim, age_dim):
        super(CrossAttention, self).__init__()
        self.q = nn.Linear(pixel_dim, pixel_dim)
        self.k = nn.Linear(age_dim, pixel_dim)
        self.v = nn.Linear(age_dim, pixel_dim)
        self.scale = torch.sqrt(torch.tensor(pixel_dim, dtype=torch.float32))

    def forward(self, pixel_features, age_features):
        # pixel_features: [batch_size, num_pixels, pixel_dim]
        # age_features: [batch_size, age_dim]
        # age_features need to be expanded to match the spatial dimension of pixel features
        batch_size, num_pixels, _ = pixel_features.size()

        # Expand age features to match the number of pixels
        age_features = age_features.unsqueeze(1).expand(batch_size, num_pixels, -1)
        print('AGE_FEATURES SHAPE:', age_features.shape)

        # Compute Q, K, V
        q = self.q(pixel_features)  # [batch_size, num_pixels, pixel_dim]
        k = self.k(age_features)    # [batch_size, num_pixels, pixel_dim]
        v = self.v(age_features)    # [batch_size, num_pixels, pixel_dim]

        print('q SHAPE:', q.shape)
        print('k SHAPE:', k.shape)
        # Cross-attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_pixels, num_pixels]
        weights = F.softmax(scores, dim=-1)

        # Output is weighted sum of values
        attended_features = torch.matmul(weights, v)
        return attended_features + pixel_features  # Residual connection

# Self-Attention Module (SA)
class SelfAttention(nn.Module):
    """
    Standard self-attention mechanism.
    """
    def __init__(self, pixel_dim):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(pixel_dim, pixel_dim * 3)  # For generating Q, K, V in one operation
        self.scale = torch.sqrt(torch.tensor(pixel_dim, dtype=torch.float32))

    def forward(self, pixel_features):
        batch_size, num_pixels, _ = pixel_features.size()
        qkv = self.qkv(pixel_features).view(batch_size, num_pixels, 3, -1)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]

        print('q self-attention', q.shape)
        print('k self-attention', k.shape)
        print(f"q tensor size: {q.size()} | Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"v tensor size: {v.size()} | Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Compute attention scores and weighted sum
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        print('SHAPE of self-att:', torch.matmul(weights, v).shape, pixel_features.shape)
        return torch.matmul(weights, v) + pixel_features  # Residual connection

# LoCI Fusion Module
class LoCIFusionModule(nn.Module):
    """
    Longitudinal Consistency-Informed (LoCI) module with cross-attention for preceding and subsequent latent spaces.
    """
    def __init__(self, pixel_dim):
        super(LoCIFusionModule, self).__init__()
        self.cross_attention = CrossAttention(pixel_dim, pixel_dim)
        self.norm1 = nn.LayerNorm(pixel_dim)
        self.norm2 = nn.LayerNorm(pixel_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(pixel_dim, pixel_dim),
            nn.ReLU(),
            nn.Linear(pixel_dim, pixel_dim)
        )

    def forward(self, p, s):
        # p and s are the latent spaces from preceding and subsequent images
        attended_p = self.cross_attention(p, s)
        attended_s = self.cross_attention(s, p)

        # Normalize and feed forward
        p_norm = self.norm1(attended_p + p)
        s_norm = self.norm2(attended_s + s)
        p_ff = self.feed_forward(p_norm)
        s_ff = self.feed_forward(s_norm)

        return p_ff, s_ff

# Global Attention Mechanism (GAM)
class GAM(nn.Module):
    """
    Global Attention Mechanism (GAM) for fusing global features.
    """
    def __init__(self, pixel_dim):
        super(GAM, self).__init__()
        self.fc1 = nn.Linear(pixel_dim, pixel_dim)
        self.fc2 = nn.Linear(pixel_dim, pixel_dim)
        self.conv = nn.Conv3d(in_channels=pixel_dim, out_channels=pixel_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fusion_condition):
        x = self.fc1(fusion_condition)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.conv(x.unsqueeze(1))  # Adding a channel dimension
        return self.sigmoid(x)

# Full Denoising Network with LoCI Fusion, Age Embedding, and GAM
class DenoisingNetwork(nn.Module):
    def __init__(self, input_shape, filters=64, age_embedding_dim=128):
        super(DenoisingNetwork, self).__init__()
        # Embedding the age information
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim)

        # Residual, Downsample, Upsample blocks (simplified for demonstration)
        self.res_block = nn.Conv3d(1, filters, kernel_size=3, padding=1)
        self.downsample = nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(filters, filters, kernel_size=3, stride=2, padding=1)

        # Self-Attention layer
        self.self_attention = SelfAttention(pixel_dim=filters)

        # LoCI Fusion Module
        self.loci_fusion = LoCIFusionModule(pixel_dim=filters)

        # Global Attention Mechanism
        self.gam = GAM(pixel_dim=filters)

        # Final output convolution
        self.final_conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=3, padding=1)

    def forward(self, p, t, s, age):
        # Embed the age information
        age_emb = self.age_embedding(age)

        # Process preceding, target, and subsequent images
        p_features = self.downsample(self.res_block(p))
        t_features = self.downsample(self.res_block(t))
        s_features = self.downsample(self.res_block(s))

        # Apply Self-Attention
        p_features = self.self_attention(p_features.view(p_features.size(0), -1, p_features.size(1)))
        s_features = self.self_attention(s_features.view(s_features.size(0), -1, s_features.size(1)))

        # LoCI Fusion
        fused_p, fused_s = self.loci_fusion(p_features, s_features)

        # Use GAM to fuse with the age information
        fusion_condition = fused_p + fused_s + t_features.view(t_features.size(0), -1, t_features.size(1))  # Simplified
        gam_output = self.gam(fusion_condition)

        # Upsample and reconstruct the image
        reconstructed = self.upsample(gam_output)
        output = self.final_conv(reconstructed)

        return output
    
class DenoisingNetworkParallel(nn.Module):
    def __init__(self, input_shape, filters=64, age_embedding_dim=128):
        super(DenoisingNetworkParallel, self).__init__()

        # GPU 0 - Embedding and Downsample
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim).to('cuda:0')
        self.res_block = nn.Conv3d(1, filters, kernel_size=3, padding=1).to('cuda:0')
        self.downsample = nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1).to('cuda:0')

        # GPU 1 - Self-Attention and LoCI Fusion
        self.self_attention = SelfAttention(pixel_dim=filters).to('cuda:1')
        self.loci_fusion = LoCIFusionModule(pixel_dim=filters).to('cuda:1')

        # GPU 2 - GAM and Final Reconstruction
        self.gam = GAM(pixel_dim=filters).to('cuda:2')
        self.upsample = nn.ConvTranspose3d(filters, filters, kernel_size=3, stride=2, padding=1).to('cuda:2')
        self.final_conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=3, padding=1).to('cuda:2')

    def forward(self, p, t, s, age):
        # Ensure inputs are on GPU 0
        p, t, s, age = p.to('cuda:0'), t.to('cuda:0'), s.to('cuda:0'), age.to('cuda:0')

        # Stage 1: Age embedding and downsampling on GPU 0
        age_emb = self.age_embedding(age)
        p_features = self.downsample(self.res_block(p))
        t_features = self.downsample(self.res_block(t))
        s_features = self.downsample(self.res_block(s))

        # Move to GPU 1 for self-attention and LoCI fusion
        p_features, s_features = p_features.to('cuda:1'), s_features.to('cuda:1')
        p_features = self.self_attention(p_features.flatten(2).permute(0, 2, 1))  # [B, N, D]
        s_features = self.self_attention(s_features.flatten(2).permute(0, 2, 1))
        fused_p, fused_s = self.loci_fusion(p_features, s_features)

        # Move to GPU 2 for GAM and reconstruction
        fused_p, fused_s = fused_p.to('cuda:2'), fused_s.to('cuda:2')
        t_features = t_features.flatten(2).permute(0, 2, 1).to('cuda:2')

        # Apply GAM
        fusion_condition = fused_p + fused_s + t_features
        gam_output = self.gam(fusion_condition)

        # Reshape and reconstruct the output on GPU 2
        reconstructed = self.upsample(gam_output.permute(0, 2, 1).view(*p.size()))  # Restore original shape
        output = self.final_conv(reconstructed)

        return output



# # Example usage
# input_shape = (64, 64, 64)  # Example 3D input shape (depth, height, width)
# filters = 64
# age_embedding_dim = 128
# batch_size = 8

# model = DenoisingNetwork(input_shape, filters, age_embedding_dim)

# # Dummy input tensors
# p = torch.randn(batch_size, 1, *input_shape)  # Preceding
# t = torch.randn(batch_size, 1, *input_shape)  # Target
# s = torch.randn(batch_size, 1, *input_shape)  # Subsequent
# age = torch.randn(batch_size, 1)  # Age information

# # Forward pass
# output = model(p, t, s, age)
# print(output.shape)

# ####################################################################################


# from torch.utils.data import DataLoader
# # Initialize the dataset
# dataset = CP(root_dir='/CP/sub-001', age_csv='/path/to/trios_sorted_by_age.csv')

# # Create a DataLoader
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# # Example: Use the Dataloader to feed data into your model
# for p, t, s, age in dataloader:
#     # p: Preceding image, t: Target image, s: Subsequent image, age: Age tensor
#     output = model(p, t, s, age)
#     print(output.shape)
