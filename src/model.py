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


class TransformerWithSelfAttention(nn.Module):
    """
    Transformer Encoder block with self-attention and feed-forward layers.
    """
    def __init__(self, pixel_dim, ff_dim=2048, num_heads=8, dropout=0.1):
        super(TransformerWithSelfAttention, self).__init__()

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads, dropout=dropout)

        # Feed-Forward Network (FFN) with two linear layers and ReLU in between
        self.ffn = nn.Sequential(
            nn.Linear(pixel_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, pixel_dim)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(pixel_dim)
        self.norm2 = nn.LayerNorm(pixel_dim)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention with residual connection and layer normalization
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feed-Forward Network with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # Residual connection
        x = self.norm2(x)

        return x

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
    
# Cross-Attention Module
class CrossAttentionWithAge(nn.Module):
    """
    Cross-attention between pixel features and age features (textual features).
    """
    def __init__(self, pixel_dim, age_dim):
        super(CrossAttentionWithAge, self).__init__()
        self.q = nn.Linear(age_dim, pixel_dim)  # Query from age features
        self.k = nn.Linear(pixel_dim, pixel_dim)  # Key from pixel features
        self.v = nn.Linear(pixel_dim, pixel_dim)  # Value from pixel features
        self.scale = torch.sqrt(torch.tensor(pixel_dim, dtype=torch.float32))

    def forward(self, pixel_features, age_features):
        # pixel_features: [batch_size, num_pixels, pixel_dim]
        # age_features: [batch_size, age_dim]
        batch_size, num_pixels, _ = pixel_features.size()

        # Compute Q, K, V
        q = self.q(age_features).unsqueeze(1)  # Expand age features for cross-attention
        k = self.k(pixel_features)
        v = self.v(pixel_features)

        # Cross-attention mechanism
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, 1, num_pixels]
        weights = F.softmax(scores, dim=-1)

        # Output is weighted sum of values
        attended_features = torch.matmul(weights, v)
        return attended_features + pixel_features  # Residual connection

# Global Attention Mechanism (GAM) with Cross-Attention for Age
class GAMWithAge(nn.Module):
    """
    Global Attention Mechanism (GAM) for fusing global features with age embedding.
    Includes both channel attention and spatial attention.
    """
    def __init__(self, pixel_dim, age_dim, reduction_ratio=4):
        super(GAMWithAge, self).__init__()

        # Channel attention (Permute + MLP)
        self.channel_mlp = nn.Sequential(
            nn.Linear(pixel_dim, pixel_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(pixel_dim // reduction_ratio, pixel_dim)
        )

        # Spatial attention (Convolutional layers for attention map)
        self.conv1 = nn.Conv3d(pixel_dim, pixel_dim, kernel_size=1)
        self.conv2 = nn.Conv3d(pixel_dim, pixel_dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Cross-attention between pixel features and age embedding
        self.cross_attention = CrossAttentionWithAge(pixel_dim=pixel_dim, age_dim=age_dim)

    def forward(self, fusion_condition, age_emb):
        # Channel Attention: Apply channel-wise attention
        permuted = fusion_condition.permute(0, 2, 1)  # [batch_size, pixel_dim, num_pixels]
        channel_attention = self.channel_mlp(permuted).permute(0, 2, 1)  # Restore original order

        # Spatial Attention: Apply spatial attention using convolution
        spatial_attention = self.sigmoid(self.conv2(F.relu(self.conv1(fusion_condition.unsqueeze(1))))).squeeze(1)

        # Cross-Attention with Age Embedding
        cross_attended_features = self.cross_attention(spatial_attention, age_emb)

        # Final fused output combines channel, spatial, and age cross-attention
        return cross_attended_features + channel_attention

# Full Denoising Network with LoCI Fusion, TransformerWithSelfAttention, Age Embedding, and GAMWithAge
class DenoisingNetwork(nn.Module):
    def __init__(self, input_shape, filters=64, age_embedding_dim=128):
        super(DenoisingNetwork, self).__init__()
        # Embedding the age information
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim)

        # Residual, Downsample, Upsample blocks (simplified for demonstration)
        self.res_block = nn.Conv3d(1, filters, kernel_size=3, padding=1)
        self.downsample = nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(filters, filters, kernel_size=3, stride=2, padding=1)

        # LoCI Fusion Module
        self.loci_fusion = LoCIFusionModule(pixel_dim=filters)

        # TransformerWithSelfAttention to get C_fused after LoCI
        self.transformer = TransformerWithSelfAttention(pixel_dim=filters)

        # Global Attention Mechanism with Age embedding
        self.gam_with_age = GAMWithAge(pixel_dim=filters, age_dim=age_embedding_dim)

        # Final output convolution
        self.final_conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=3, padding=1)

    def forward(self, p, t, s, age):
        # Embed the age information
        age_emb = self.age_embedding(age)

        # Process preceding, target, and subsequent images
        p_features = self.downsample(self.res_block(p))
        t_features = self.downsample(self.res_block(t))
        s_features = self.downsample(self.res_block(s))

        # LoCI Fusion
        fused_p, fused_s = self.loci_fusion(p_features, s_features)

        # Create C_fused using Transformer with Self-Attention
        c_fused = self.transformer(fused_p + fused_s)

        # Use GAM with Cross-Attention for Age to fuse with the age information
        gam_output = self.gam_with_age(c_fused, age_emb)

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
