import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        """
        Compute sinusoidal embeddings for time steps.
        t: Tensor of shape [batch_size], values between 0 and 1.
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # Shape: [batch_size, embedding_dim]

# Age Embedding
class AgeEmbedding(nn.Module):
    """
    Embeds the age input as textual features.
    """
    def __init__(self, embedding_dim):
        super(AgeEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, age):
        # age is expected to be a tensor of shape [batch_size, 1]
        return self.embedding(age)

# Define the RB, DS, and US blocks separately
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, filters):
        super(DownsampleBlock, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class UpsampleBlock(nn.Module):
    def __init__(self, filters):
        super(UpsampleBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(filters, filters, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)

# Global Attention Mechanism (GAM)
class GAM(nn.Module):
    """ Adapted from: https://github.com/dengbuqi/GAM_Pytorch/blob/main/CAM.py """
    def __init__(self, in_channels, out_channels, age_embedding_dim=128, rate=4):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)

        # Channel Attention Components
        self.linear1 = nn.Linear(in_channels, inchannel_rate)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels)
        
        # Spatial Attention Components
        self.conv1 = nn.Conv3d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
        self.conv2 = nn.Conv3d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')

        self.norm1 = nn.BatchNorm3d(inchannel_rate)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.sigmoid = nn.Sigmoid()

        # Age Embedding Component
        self.age_fc = nn.Linear(age_embedding_dim, in_channels)

    def forward(self, x, age_emb):
        b, c, d, h, w = x.shape

        # Channel Attention
        x_permute = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).reshape(b, d, h, w, c)
        x_channel_att = x_att_permute.permute(0, 4, 1, 2, 3)
        x = x * x_channel_att

        # Age-based Modulation
        age_modulation = self.age_fc(age_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        age_modulation = age_modulation.expand(-1, -1, d, h, w)
        x = x * age_modulation

        # Spatial Attention
        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))
        out = x * x_spatial_att

        return out

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
        # pixel_features: [batch_size, channels, depth, height, width] for 3D CNN
        # age_features: [batch_size, age_dim]

        # Reshape pixel_features to [batch_size, num_pixels, channels]
        batch_size, channels, depth, height, width = pixel_features.size()
        num_pixels = depth * height * width

        # Reshape pixel features from [batch_size, channels, depth, height, width] to [batch_size, num_pixels, channels]
        pixel_features = pixel_features.view(batch_size, channels, num_pixels).permute(0, 2, 1)  # [batch_size, num_pixels, channels]

        # Reshape age_features to [batch_size, age_dim] (make sure age_features has 2 dimensions)
        age_features = age_features.view(batch_size, -1)  # Flatten to 2D if necessary

        # Expand age features to match the number of pixels
        age_features = age_features.unsqueeze(1).expand(batch_size, num_pixels, age_features.size(-1))

        # Compute Q, K, V
        q = self.q(pixel_features)  # [batch_size, num_pixels, pixel_dim]
        k = self.k(age_features)    # [batch_size, num_pixels, pixel_dim]
        v = self.v(age_features)    # [batch_size, num_pixels, pixel_dim]

        # Cross-attention: Ensure Q and K have matching dimensions for multiplication
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_pixels, num_pixels]
        weights = F.softmax(scores, dim=-1)

        # Output is weighted sum of values
        attended_features = torch.matmul(weights, v)

        # Residual connection: Add the attended features to the original pixel features
        return attended_features + pixel_features

class TransformerWithSelfAttention(nn.Module):
    """
    Transformer Encoder block with self-attention and feed-forward layers.
    Expects input of shape [seq_len, batch_size, embed_dim].
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
        """
        x: Tensor of shape [seq_len, batch_size, embed_dim]
        """
        # Self-Attention with residual connection and layer normalization
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feed-Forward Network with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)  # Residual connection
        x = self.norm2(x)

        return x  # Output shape: [seq_len, batch_size, embed_dim]


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
class LoCIFusionModuleFIRST(nn.Module):
    def __init__(self, pixel_dim, num_heads=8):
        super(LoCIFusionModule, self).__init__()
        self.num_layers = 3  # Number of cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        self.layer_norms_p = nn.ModuleList()
        self.layer_norms_s = nn.ModuleList()
        self.feed_forward_p = nn.ModuleList()
        self.feed_forward_s = nn.ModuleList()

        for _ in range(self.num_layers):
            # Cross-attention layers for p_features and s_features
            self.cross_attention_layers.append(
                nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads)
            )
            # Layer Normalization
            self.layer_norms_p.append(nn.LayerNorm(pixel_dim))
            self.layer_norms_s.append(nn.LayerNorm(pixel_dim))
            # Feed-forward networks
            self.feed_forward_p.append(
                nn.Sequential(
                    nn.Linear(pixel_dim, pixel_dim),
                    nn.ReLU(),
                    nn.Linear(pixel_dim, pixel_dim),
                )
            )
            self.feed_forward_s.append(
                nn.Sequential(
                    nn.Linear(pixel_dim, pixel_dim),
                    nn.ReLU(),
                    nn.Linear(pixel_dim, pixel_dim),
                )
            )

    def forward(self, p_features, s_features):
        # Reshape to [seq_len, batch_size, embed_dim] for MultiheadAttention
        batch_size, channels, D, H, W = p_features.shape
        seq_len = D * H * W

        # Flatten spatial dimensions and permute
        p_features = p_features.view(batch_size, channels, seq_len).permute(2, 0, 1)
        s_features = s_features.view(batch_size, channels, seq_len).permute(2, 0, 1)

        for i in range(self.num_layers):
            # Cross-attention from p to s
            # Query: p_features, Key/Value: s_features
            attn_output_p, _ = self.cross_attention_layers[i](p_features, s_features, s_features)
            p_features = self.layer_norms_p[i](p_features + attn_output_p)
            p_features = p_features + self.feed_forward_p[i](p_features)

            # Cross-attention from s to p
            # Query: s_features, Key/Value: p_features
            attn_output_s, _ = self.cross_attention_layers[i](s_features, p_features, p_features)
            s_features = self.layer_norms_s[i](s_features + attn_output_s)
            s_features = s_features + self.feed_forward_s[i](s_features)

        # After LoCI fusion, obtain context-aware consistency features
        C_Pi = p_features
        C_Si = s_features

        # Compute Mean Squared Error (MSE) between C_Pi and C_Si for the loss function
        # (This should be done outside this module, in your training loop or loss function)

        # Optionally, you can average the features to get fused features
        # fused_features = (C_Pi + C_Si) / 2

        # Reshape back to [batch_size, channels, D, H, W]
        # fused_features = fused_features.permute(1, 2, 0).view(batch_size, channels, D, H, W)

        return C_Pi, C_Si  
    
class LoCIFusionModule(nn.Module):
    def __init__(self, pixel_dim, num_heads=8):
        super(LoCIFusionModule, self).__init__()
        self.num_layers = 3  # Number of cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        self.layer_norms_p = nn.ModuleList()
        self.layer_norms_s = nn.ModuleList()
        self.feed_forward_p = nn.ModuleList()
        self.feed_forward_s = nn.ModuleList()

        for _ in range(self.num_layers):
            # Cross-attention layers for p_features and s_features
            self.cross_attention_layers.append(
                nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads)
            )
            # Layer Normalization
            self.layer_norms_p.append(nn.LayerNorm(pixel_dim))
            self.layer_norms_s.append(nn.LayerNorm(pixel_dim))
            # Feed-forward networks
            self.feed_forward_p.append(
                nn.Sequential(
                    nn.Linear(pixel_dim, pixel_dim),
                    nn.ReLU(),
                    nn.Linear(pixel_dim, pixel_dim),
                )
            )
            self.feed_forward_s.append(
                nn.Sequential(
                    nn.Linear(pixel_dim, pixel_dim),
                    nn.ReLU(),
                    nn.Linear(pixel_dim, pixel_dim),
                )
            )

    def forward(self, p_features, s_features):
        # p_features and s_features: [seq_len, batch_size, pixel_dim]
        for i in range(self.num_layers):
            # Cross-attention from p to s
            attn_output_p, _ = self.cross_attention_layers[i](p_features, s_features, s_features)
            p_features = self.layer_norms_p[i](p_features + attn_output_p)
            p_features = p_features + self.feed_forward_p[i](p_features)

            # Cross-attention from s to p
            attn_output_s, _ = self.cross_attention_layers[i](s_features, p_features, p_features)
            s_features = self.layer_norms_s[i](s_features + attn_output_s)
            s_features = s_features + self.feed_forward_s[i](s_features)

        # After LoCI fusion, obtain context-aware consistency features
        C_Pi = p_features  # [seq_len, batch_size, pixel_dim]
        C_Si = s_features  # [seq_len, batch_size, pixel_dim]

        return C_Pi, C_Si



class LoCIFusionModuleV2(nn.Module):
    def __init__(self, pixel_dim, num_heads=8, ff_dim=256, dropout=0.1):
        super(LoCIFusionModule, self).__init__()

        # Cross-Attention Layers
        self.cross_attention_1 = nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attention_2 = nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attention_3 = nn.MultiheadAttention(embed_dim=pixel_dim, num_heads=num_heads, dropout=dropout)

        # Feed-forward network after each attention layer
        self.feed_forward = nn.Sequential(
            nn.Linear(pixel_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, pixel_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization after each attention and feed-forward step
        self.layer_norm_1 = nn.LayerNorm(pixel_dim)
        self.layer_norm_2 = nn.LayerNorm(pixel_dim)
        self.layer_norm_3 = nn.LayerNorm(pixel_dim)

        # Mean Squared Error (MSE) loss for consistency between P_t and S_t
        self.mse_loss = nn.MSELoss()

    def forward(self, p_features, s_features):
        # Cross-attention for the first stage
        p_q = s_k = s_v = s_features  # Query: subsequent, Key/Value: subsequent
        s_q = p_k = p_v = p_features  # Query: preceding, Key/Value: preceding

        p_fused, _ = self.cross_attention_1(p_q, p_k, p_v)
        s_fused, _ = self.cross_attention_1(s_q, s_k, s_v)

        # Residual connection and normalization
        p_fused = self.layer_norm_1(p_fused + p_features)
        s_fused = self.layer_norm_1(s_fused + s_features)

        # Cross-attention for the second stage
        p_q, s_k, s_v = s_fused, s_fused, s_fused
        s_q, p_k, p_v = p_fused, p_fused, p_fused

        p_fused, _ = self.cross_attention_2(p_q, p_k, p_v)
        s_fused, _ = self.cross_attention_2(s_q, s_k, s_v)

        # Residual connection and normalization
        p_fused = self.layer_norm_2(p_fused + p_features)
        s_fused = self.layer_norm_2(s_fused + s_features)

        # Cross-attention for the third stage
        p_q, s_k, s_v = s_fused, s_fused, s_fused
        s_q, p_k, p_v = p_fused, p_fused, p_fused

        p_fused, _ = self.cross_attention_3(p_q, p_k, p_v)
        s_fused, _ = self.cross_attention_3(s_q, s_k, s_v)

        # Residual connection and normalization
        p_fused = self.layer_norm_3(p_fused + p_features)
        s_fused = self.layer_norm_3(s_fused + s_features)

        # Consistency Loss between fused P_t and S_t
        mse_loss = self.mse_loss(p_fused, s_fused)

        # Output fused subsequent feature C_Si and the MSE loss
        return s_fused, mse_loss  # Output C_Si as the fused feature

    
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


# Fusion for guiding DPM
class FusionModule(nn.Module):
    def __init__(self, in_channels=1, filters=64, age_embedding_dim=128, num_repeats=4):
        super(FusionModule, self).__init__()

        # Embedding the age information
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim)

        # Define the repeated blocks for residual, downsample, LoCI Fusion, and Transformer blocks
        self.residual_blocks = nn.ModuleList()   # For Residual Blocks
        self.downsample_blocks = nn.ModuleList() # For Downsampling
        self.loci_fusion_blocks = nn.ModuleList()  # For LoCI Fusion
        self.self_attention_blocks = nn.ModuleList() # For Transformer Self-Attention

        current_channels = in_channels
        for _ in range(num_repeats):
            # Add the Residual Block
            self.residual_blocks.append(ResidualBlock(in_channels=current_channels, filters=filters))

            # Add the Downsample Block
            self.downsample_blocks.append(DownsampleBlock(filters=filters))

            # Add the LoCI Fusion Block
            self.loci_fusion_blocks.append(LoCIFusionModule(pixel_dim=filters))

            # Add Transformer Self-Attention Block
            self.self_attention_blocks.append(TransformerWithSelfAttention(pixel_dim=filters))

            # Update the current channels to be the number of filters
            current_channels = filters

    def forward(self, p, s, age):
        # Embed the age information
        age_emb = self.age_embedding(age)

        # Process preceding and subsequent images through the RB+DS stages
        p_features = p
        s_features = s

        c_pred_p_list = []
        c_pred_s_list = []

        # Store spatial dimensions for reshaping
        spatial_dims = None

        # Sequentially apply Residual, Downsampling, LoCI Fusion, and Transformer Self-Attention blocks
        for i in range(len(self.residual_blocks)):
            # Apply Residual Block
            p_features = self.residual_blocks[i](p_features)
            s_features = self.residual_blocks[i](s_features)

            # Apply Downsampling
            p_features = self.downsample_blocks[i](p_features)
            s_features = self.downsample_blocks[i](s_features)

            # Store spatial dimensions after downsampling
            batch_size, channels, depth, height, width = p_features.size()
            spatial_dims = (depth, height, width)

            # Flatten spatial dimensions and permute for LoCI Fusion
            p_flat = p_features.view(batch_size, channels, -1).permute(2, 0, 1)  # [seq_len, batch_size, channels]
            s_flat = s_features.view(batch_size, channels, -1).permute(2, 0, 1)  # [seq_len, batch_size, channels]

            # Apply LoCI Fusion on preceding and subsequent features
            C_Pi, C_Si = self.loci_fusion_blocks[i](p_flat, s_flat)
            c_pred_p_list.append(C_Pi)
            c_pred_s_list.append(C_Si)

            # Apply Transformer Self-Attention to fused features (using C_Si)
            c_fused = self.self_attention_blocks[i](C_Si)

            # Reshape c_fused back to [batch_size, channels, depth, height, width]
            c_fused = c_fused.permute(1, 2, 0).view(batch_size, channels, *spatial_dims)

            # Update p_features and s_features for next iteration if needed
            # In your case, you may proceed directly depending on your architecture

        return c_fused, c_pred_p_list, c_pred_s_list



class GAMUNet(nn.Module):
    def __init__(self, in_channels=1, filters=64, age_embedding_dim=128, time_embedding_dim=128, num_repeats=4):
        super(GAMUNet, self).__init__()

        # Embedding the age and time information
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim)
        self.time_embedding = SinusoidalTimeEmbedding(embedding_dim=time_embedding_dim)

        # Total embedding dimension
        self.total_embedding_dim = age_embedding_dim + time_embedding_dim

        # Initial convolution to incorporate embeddings
        self.initial_conv = nn.Conv3d(in_channels + self.total_embedding_dim, filters, kernel_size=3, padding=1)

        # Encoder blocks: Residual Blocks followed by Downsampling
        self.encoder_residual_blocks = nn.ModuleList()
        self.encoder_downsample_blocks = nn.ModuleList()

        # Decoder blocks: GAM, Residual Blocks, Upsampling
        self.GAM_blocks = nn.ModuleList()
        self.decoder_residual_blocks = nn.ModuleList()
        self.decoder_upsample_blocks = nn.ModuleList()

        current_channels = filters
        for _ in range(num_repeats):
            # Encoder Residual Block
            self.encoder_residual_blocks.append(ResidualBlock(in_channels=current_channels, filters=filters))

            # Encoder Downsampling Block
            self.encoder_downsample_blocks.append(DownsampleBlock(filters=filters))

            current_channels = filters  # Update current_channels

        for _ in range(num_repeats):
            # Global Attention Mechanism (GAM)
            self.GAM_blocks.append(GAM(filters, filters, age_embedding_dim))

            # Residual Block for Decoder with input from concatenation
            self.decoder_residual_blocks.append(ResidualBlock(in_channels=filters * 2, filters=filters))

            # Upsample Block for Decoder
            self.decoder_upsample_blocks.append(UpsampleBlock(filters=filters))

        # Final output convolution to predict the noise
        self.final_conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=3, padding=1)

    def forward(self, t_input, c_fused, age, t):
        # Embed the age and time information
        age_emb = self.age_embedding(age)  # Shape: [batch_size, age_embedding_dim]
        time_emb = self.time_embedding(t)  # Shape: [batch_size, time_embedding_dim]

        # Combine embeddings
        emb = torch.cat([age_emb, time_emb], dim=-1)  # Shape: [batch_size, total_embedding_dim]

        # Reshape embeddings to match spatial dimensions
        emb = emb[:, :, None, None, None]  # Shape: [batch_size, total_embedding_dim, 1, 1, 1]
        emb = emb.expand(-1, -1, t_input.shape[2], t_input.shape[3], t_input.shape[4])  # Match spatial dimensions

        # Concatenate embeddings with t_input
        x = torch.cat([t_input, emb], dim=1)  # Concatenate along channel dimension

        # Initial convolution
        x = self.initial_conv(x)

        # Encoder path
        encoder_outputs = []  # To store encoder outputs for skip connections

        for i in range(len(self.encoder_residual_blocks)):
            # Apply Residual Block
            x = self.encoder_residual_blocks[i](x)

            # Save the features for skip connections
            encoder_outputs.append(x)

            # Apply Downsampling
            x = self.encoder_downsample_blocks[i](x)

        # Reverse the encoder outputs for easy access in decoder
        encoder_outputs = encoder_outputs[::-1]  # Reverse the list

        # Decoder path
        for i in range(len(self.GAM_blocks)):
            # Apply Global Attention Mechanism with Age Embedding
            gam_output = self.GAM_blocks[i](c_fused, age_emb)

            # Retrieve the corresponding encoder output for skip connection
            skip_connection = encoder_outputs[i]

            # Concatenate encoder output (skip connection) with GAM output
            concatenated_output = torch.cat([skip_connection, gam_output], dim=1)  # Concatenate along channel dimension

            # Apply Residual Block for Decoder
            concatenated_output = self.decoder_residual_blocks[i](concatenated_output)

            # Apply Upsample Block for Decoder
            x = self.decoder_upsample_blocks[i](concatenated_output)

        # Final convolution to predict the noise
        predicted_noise = self.final_conv(x)
        return predicted_noise


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
    

# Define the diffusion model guided by the fusion module
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DPM(nn.Module):
    def __init__(self, fusion_model, nn_model, betas, n_T, device):
        super(DPM, self).__init__()
        self.fusion_model = fusion_model.to(device)
        self.nn_model = nn_model.to(device)

        # Register buffers for diffusion schedules
        schedules = ddpm_schedules(betas[0], betas[1], n_T)
        for k, v in schedules.items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, p, t, s, age, lambda_fusion=0.6):
        """
        Training forward pass.
        """
        batch_size = t.shape[0]

        # Generate random timesteps
        _ts = torch.randint(1, self.n_T, (batch_size,), device=self.device)

        # Add noise to the target image
        noise = torch.randn_like(t).to(self.device)
        sqrtab_t = self.sqrtab[_ts].view(-1, 1, 1, 1, 1)
        sqrtmab_t = self.sqrtmab[_ts].view(-1, 1, 1, 1, 1)
        x_t = sqrtab_t * t + sqrtmab_t * noise  # Noisy image

        # Get fused features and context-aware consistency features from the fusion model
        c_fused, c_pred_p_list, c_pred_s_list = self.fusion_model(p, s, age)

        # Predict the noise using GAMUNet
        predicted_noise = self.nn_model(x_t, c_fused, age, _ts / self.n_T)

        # Compute diffusion loss
        l_diff = self.loss_mse(noise, predicted_noise)

        # Compute fusion loss (average over all levels)
        l_fusion = 0.0
        for c_pred_p, c_pred_s in zip(c_pred_p_list, c_pred_s_list):
            l_fusion += self.loss_mse(c_pred_p, c_pred_s)
        l_fusion = l_fusion / len(c_pred_p_list)  # Average over levels

        # Compute total loss
        loss = l_diff + lambda_fusion * l_fusion

        return loss

    def sample(self, p, s, age, skip_step=80):
        """
        Sampling method to generate the denoised image starting from random noise.
        Performs denoising with skip steps.
        """
        self.nn_model.eval()
        with torch.no_grad():
            c_fused, _, _ = self.fusion_model(p, s, age)

            n_sample = c_fused.shape[0]
            size = c_fused.shape[1:]

            x_i = torch.randn(n_sample, *size).to(self.device)  # x_T ~ N(0, 1)

            timesteps = list(range(self.n_T, 0, -skip_step))
            if timesteps[-1] != 0:
                timesteps.append(0)  # Ensure we reach timestep 0

            for i in timesteps:
                t_is = torch.full((n_sample,), i / self.n_T, device=self.device)

                sqrtab_t = self.sqrtab[i]
                oneover_sqrta_t = self.oneover_sqrta[i]
                mab_over_sqrtmab_t = self.mab_over_sqrtmab[i]
                sqrt_beta_t = self.sqrt_beta_t[i]

                z = torch.randn(n_sample, *size).to(self.device) if i > 0 else 0

                predicted_noise = self.nn_model(x_i, c_fused, age, t_is)

                x_i = (
                    oneover_sqrta_t * (x_i - predicted_noise * mab_over_sqrtmab_t)
                    + sqrt_beta_t * z
                )

            return x_i
