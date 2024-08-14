import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm3d(filters)
        self.conv2 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm3d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DownSample3D(nn.Module):
    def __init__(self, filters):
        super(DownSample3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SelfAttention3D(nn.Module):
    def __init__(self, filters):
        super(SelfAttention3D, self).__init__()
        self.q = nn.Linear(filters, filters)
        self.k = nn.Linear(filters, filters)
        self.v = nn.Linear(filters, filters)
        self.scale = torch.sqrt(torch.tensor(filters, dtype=torch.float32))

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

class CrossAttention3D(nn.Module):
    def __init__(self, filters):
        super(CrossAttention3D, self).__init__()
        self.q = nn.Linear(filters, filters)
        self.k = nn.Linear(filters, filters)
        self.v = nn.Linear(filters, filters)
        self.scale = torch.sqrt(torch.tensor(filters, dtype=torch.float32))

    def forward(self, q_input, kv_input):
        q = self.q(q_input)
        k = self.k(kv_input)
        v = self.v(kv_input)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

class GAM3D(nn.Module):
    def __init__(self, filters, reduction_ratio=4):
        super(GAM3D, self).__init__()
        reduced_filters = filters // reduction_ratio
        self.linear1 = nn.Linear(filters, reduced_filters)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(reduced_filters, filters)
        self.conv = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        x = self.relu(self.linear1(x))
        x = self.linear2(x).view(batch_size, depth, height, width, channels).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        return self.sigmoid(x) * x

class LoCIFusionModule3D(nn.Module):
    def __init__(self, filters):
        super(LoCIFusionModule3D, self).__init__()
        self.self_attention = SelfAttention3D(filters)
        self.cross_attention = CrossAttention3D(filters)
        self.norm1 = nn.LayerNorm(filters)
        self.norm2 = nn.LayerNorm(filters)
        self.feed_forward = nn.Sequential(
            nn.Linear(filters, filters),
            nn.ReLU(inplace=True),
            nn.Linear(filters, filters)
        )

    def forward(self, p, s):
        sa_output = self.self_attention(p)
        ca_output = self.cross_attention(p, s)
        sa_output = self.norm1(sa_output + p)
        ca_output = self.norm2(ca_output + s)
        ff_output = self.feed_forward(sa_output + ca_output)
        return ff_output, sa_output, ca_output

def build_model_3d(input_shape, filters=64, n_loci_modules=4, reduction_ratio=4, noise_levels=(1e-4, 5e-3), lambda_loci=0.6):
    # Assuming input_shape is (channels, depth, height, width)
    preceding = torch.randn((1, *input_shape))  # Example input tensor
    target = torch.randn((1, *input_shape))
    subsequent = torch.randn((1, *input_shape))

    # Initial processing layers
    rb1 = ResidualBlock3D(filters)
    ds1 = DownSample3D(filters)
    x1 = ds1(rb1(preceding))

    rb2 = ResidualBlock3D(filters)
    ds2 = DownSample3D(filters)
    x2 = ds2(rb2(target))

    rb3 = ResidualBlock3D(filters)
    ds3 = DownSample3D(filters)
    x3 = ds3(rb3(subsequent))

    # Apply multiple LoCI Fusion modules
    loci_outputs_p = []
    loci_outputs_s = []
    loci_fusion_module = LoCIFusionModule3D(filters)

    for _ in range(n_loci_modules):
        loci_fusion, loci_output_p, loci_output_s = loci_fusion_module(x1, x3)
        loci_outputs_p.append(loci_output_p)
        loci_outputs_s.append(loci_output_s)

    # Global attention mechanism
    gam = GAM3D(filters, reduction_ratio=reduction_ratio)
    gam_output = gam(loci_fusion)

    # Upsample to match the initial input resolution
    upsample = nn.ConvTranspose3d(filters, filters, kernel_size=3, stride=2, padding=1)
    upsample_output = upsample(gam_output)

    rb4 = ResidualBlock3D(filters)
    output = nn.Conv3d(filters, 1, kernel_size=3, padding=1)(rb4(upsample_output))

    model = nn.Module()
    model.output = output
    model.loci_outputs_p = loci_outputs_p
    model.loci_outputs_s = loci_outputs_s

    # Add noise schedule for diffusion
    noise_schedule = torch.linspace(noise_levels[0], noise_levels[1], 1000)

    return model, noise_schedule

def try_model():
    input_shape = (1, 64, 64, 64)  # Example 3D input shape
    model, noise_schedule = build_model_3d(input_shape)
    print(model)

def denoise(model, noisy_input, noise_schedule, skip_steps=80):
    total_steps = len(noise_schedule)
    step_interval = total_steps // skip_steps
    current_input = noisy_input

    for step in range(0, total_steps, step_interval):
        noise_level = noise_schedule[step]
        # Perform model inference at the current noise level
        # Adjust the input for model.forward() as needed
        current_input = model.output  # Assuming the output of the model is the denoised result

    return current_input

if __name__ == '__main__':
    try_model()
    # Example usage of the denoise function
    input_shape = (1, 64, 64, 64)  # Example 3D input shape
    noisy_input = torch.randn(input_shape)  # Replace with actual noisy input
    model, noise_schedule = build_model_3d(input_shape)
    denoised_output = denoise(model, noisy_input, noise_schedule)
