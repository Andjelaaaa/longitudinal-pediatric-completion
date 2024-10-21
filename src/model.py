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

# Define the RB, DS, and US blocks separately
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Residual connection
        return F.relu(out)

class DownsampleBlock(nn.Module):
    def __init__(self, filters):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv3d(filters, filters, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpsampleBlock(nn.Module):
    def __init__(self, filters):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(filters, filters, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.conv(x)

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
# LoCI Fusion Module
class LoCIFusionModule(nn.Module):
    """
    Longitudinal Consistency-Informed (LoCI) module with cross-attention for preceding and subsequent latent spaces.
    """
    def __init__(self, pixel_dim):
        super(LoCIFusionModule, self).__init__()
        self.cross_attention = CrossAttention(pixel_dim, pixel_dim)  # Using standard CrossAttention
        self.norm1 = nn.LayerNorm(pixel_dim)
        self.norm2 = nn.LayerNorm(pixel_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(pixel_dim, pixel_dim),
            nn.ReLU(),
            nn.Linear(pixel_dim, pixel_dim)
        )

    def forward(self, p, s):
        # p and s are the latent spaces from preceding and subsequent images
        attended_p = self.cross_attention(p, s)  # Cross-attention between preceding and subsequent features
        attended_s = self.cross_attention(s, p)  # Cross-attention between subsequent and preceding features

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

class GAMWithAge(nn.Module):
    """
    Global Attention Mechanism (GAM) for fusing global features with age embedding.
    Includes both channel attention and spatial attention, with support for fusion condition and skip connection.
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
        """
        fusion_condition: the concatenated input features (e.g., noisy target, C_fused).
        age_emb: the age embedding features.
        """
        # 1. **Skip connection**: Store the original fusion condition for residual connection
        skip_connection = fusion_condition

        # 2. **Channel Attention**: Apply channel-wise attention
        permuted = fusion_condition.permute(0, 2, 1)  # [batch_size, pixel_dim, num_pixels]
        channel_attention = self.channel_mlp(permuted).permute(0, 2, 1)  # Restore original order

        # 3. **Spatial Attention**: Apply spatial attention using convolution
        spatial_attention = self.sigmoid(self.conv2(F.relu(self.conv1(fusion_condition.unsqueeze(1))))).squeeze(1)

        # 4. **Cross-Attention with Age Embedding**: Fuse spatial attention with the age embedding
        cross_attended_features = self.cross_attention(spatial_attention, age_emb)

        # 5. **Final Fusion**: Combine channel attention, spatial attention, and skip connection
        # Here, we combine the original input (skip connection), cross-attended features, and channel attention.
        final_fusion = skip_connection + cross_attended_features + channel_attention

        return final_fusion

# Full Denoising Network with LoCI Fusion, TransformerWithSelfAttention, Age Embedding, and GAMWithAge
class DenoisingNetwork(nn.Module):
    def __init__(self, input_shape, in_channels=1, filters=64, age_embedding_dim=128, num_repeats=4):
        super(DenoisingNetwork, self).__init__()

        # Embedding the age information
        self.age_embedding = AgeEmbedding(embedding_dim=age_embedding_dim)

        # Define the repeated blocks for residual, downsample, LoCI Fusion, and Transformer blocks
        self.res_downsample_blocks = nn.ModuleList()
        self.loci_fusion_blocks = nn.ModuleList()
        self.transformer_blocks = nn.ModuleList()

        current_channels = in_channels
        for _ in range(num_repeats):
            self.res_downsample_blocks.append(nn.Sequential(
                ResidualBlock(in_channels=current_channels, filters=filters),  # Residual block with in_channels and filters
                DownsampleBlock(filters=filters)  # Downsample block with filters
            ))
            self.loci_fusion_blocks.append(LoCIFusionModule(pixel_dim=filters))
            self.transformer_blocks.append(TransformerWithSelfAttention(pixel_dim=filters))
            
            # After downsampling, the number of channels may remain the same, but spatial resolution reduces.
            current_channels = filters  # Keep filters the same for subsequent layers

        # Define the GAM, residual, and upsampling blocks for decoding
        self.GAM_blocks = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for _ in range(num_repeats):
            self.GAM_blocks.append(GAMWithAge(filters, age_embedding_dim))  # Global Attention Mechanism (GAM)
            self.residual_blocks.append(ResidualBlock(in_channels=filters * 2, filters=filters))  # Residual Block with input from concatenation
            self.upsample_blocks.append(UpsampleBlock(filters=filters))  # Upsample Block with filters

        # Final output convolution
        self.final_conv = nn.Conv3d(in_channels=filters * 2, out_channels=1, kernel_size=3, padding=1)  # Filters doubled for concatenation

    def forward(self, p, t, s, age):
        # Embed the age information
        age_emb = self.age_embedding(age)

        # Process preceding, target, and subsequent images through the RB+DS stages
        p_features = p
        t_features = t
        s_features = s

        for i in range(len(self.res_downsample_blocks)):
            # Apply Residual + Downsample blocks
            p_features = self.res_downsample_blocks[i](p_features)
            t_features = self.res_downsample_blocks[i](t_features)
            s_features = self.res_downsample_blocks[i](s_features)

            # Apply LoCI Fusion on preceding and subsequent features
            fused_p, fused_s = self.loci_fusion_blocks[i](p_features, s_features)

            # Apply TransformerWithSelfAttention to fused features
            c_fused = self.transformer_blocks[i](fused_p + fused_s)

        # Apply the decoding path: GAM, Residual Block, and Upsample
        for i in range(len(self.GAM_blocks)):
            # Apply Global Attention Mechanism with Age Embedding
            gam_output = self.GAM_blocks[i](c_fused, age_emb)

            # Concatenate noisy target (t_features) with GAM output for this iteration
            concatenated_output = torch.cat([t_features, gam_output], dim=1)

            # Apply Residual Block
            concatenated_output = self.residual_blocks[i](concatenated_output)

            # Apply Upsample Block
            c_fused = self.upsample_blocks[i](concatenated_output)

        # Final convolution to reconstruct the image
        output = self.final_conv(c_fused)
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
