import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeStepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeStepEmbedding, self).__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, t):
        # Sinusoidal position embedding
        half_dim = self.fc.in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.fc(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ndim, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.ndim = ndim

        match self.ndim:
            case 1:
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.norm = nn.BatchNorm1d(out_channels)
            case 2:
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.norm = nn.BatchNorm2d(out_channels)
            case 3:
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.norm = nn.BatchNorm3d(out_channels)
            case _:
                raise ValueError(f"Unsupported ndim: {self.ndim}")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, features_dims, ndim):
        super(Encoder, self).__init__()
        self.ndim = ndim
        layers = []
        for i in range(len(features_dims) - 1):
            layers.append(ConvBlock(in_channels if i == 0 else features_dims[i], features_dims[i + 1], self.ndim))
            match self.ndim:
                case 1:
                    layers.append(nn.MaxPool1d(2))
                case 2:
                    layers.append(nn.MaxPool2d(2))
                case 3:
                    layers.append(nn.MaxPool3d(2))
                case _:
                    raise ValueError(f"Unsupported ndim: {self.ndim}")
        self.enc_blocks = nn.ModuleList(layers)

    def forward(self, x):
        features = []
        for layer in self.enc_blocks:
            x = layer(x)
            features.append(x)
        return features


class CrossAttention(nn.Module):
    def __init__(self, dim, ndim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.ndim = ndim

    def forward(self, x, cond):
        # Reshape to batch, channels, height, width, depth -> batch, weight * width * depth, channels
        b, c, *spatial_dims = x.shape
        x = x.view(b, c, -1).transpose(1, 2) # [B, N, C]
        cond = cond.view(b, c, -1).transpose(1, 2) # [B, M, C]

        q = self.query(x) # Query
        k = self.key(cond) # Key
        v = self.value(cond) # Value

        # Attention
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention
        out = torch.bmm(attn_weights, v).transpose(1, 2)

        # Reshape back to batch, channels, height, width, depth
        match self.ndim:
            case 1:
                return out.view(b, c, spatial_dims[0])
            case 2:
                return out.view(b, c, spatial_dims[0], spatial_dims[1])
            case 3:
                return out.view(b, c, spatial_dims[0], spatial_dims[1], spatial_dims[2])
            case _:
                raise ValueError(f"Unsupported ndim: {self.ndim}")


class Decoder(nn.Module):
    def __init__(self, feature_dims, out_channels, ndim):
        super(Decoder, self).__init__()
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ndim = ndim
        for i in range(len(feature_dims) - 1, 0, -1):
            match self.ndim:
                case 1:
                    self.upsamples.append(nn.ConvTranspose1d(feature_dims[i], feature_dims[i-1], kernel_size=2, stride=2))
                case 2:
                    self.upsamples.append(nn.ConvTranspose2d(feature_dims[i], feature_dims[i-1], kernel_size=2, stride=2))
                case 3:
                    self.upsamples.append(nn.ConvTranspose3d(feature_dims[i], feature_dims[i-1], kernel_size=2, stride=2))
                case _:
                    raise ValueError(f"Unsupported ndim: {self.ndim}")
            self.decoders.append(ConvBlock(feature_dims[i], feature_dims[i-1], self.ndim))
        match self.ndim:
            case 1:
                self.final_conv = nn.Conv1d(feature_dims[0], out_channels, kernel_size=1)
            case 2:
                self.final_conv = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)
            case 3:
                self.final_conv = nn.Conv3d(feature_dims[0], out_channels, kernel_size=1)
            case _:
                raise ValueError(f"Unsupported ndim: {self.ndim}")

    def forward(self, x, enc_features, cond):
        for i in range(len(self.upsamples)):
            x = self.upsamples[i](x)
            x = torch.cat([x, enc_features[-i-1]], dim=1)
            x = self.decoders[i](x)
            if cond is not None:
                x = CrossAttention(x.size(1), self.ndim)(x, cond)
        return self.final_conv(x)


class MedSegDiff(nn.Module):
    def __init__(self, in_channels, out_channels, features_dims, ndim):
        super(MedSegDiff, self).__init__()
        self.time_embedding = TimeStepEmbedding(features_dims[0])
        self.encoder = Encoder(in_channels, features_dims, ndim)
        self.ff_parser = ConvBlock(features_dims[-1], features_dims[-1], ndim) # Intermediate refinement
        self.decoder = Decoder(features_dims, out_channels, ndim)
        self.ndim = ndim

    def forward(self, x, cond_image, time_step):
        # Step 1: Embed the time step
        t_emb = self.time_embedding(time_step)
        # Add time embedding to the input
        match self.ndim:
            case 1:
                x += t_emb[:, :, None]
            case 2:
                x += t_emb[:, :, None, None]
            case 3:
                x += t_emb[:, :, None, None, None]

        # Step 2: Encode the noisy input mask with skip connections
        enc_features = self.encoder(x)

        # Step 3: Refine with FF-Parser
        x = self.ff_parser(enc_features[-1])

        # Step 4: Decode with cross-attention conditioning on the input image
        output = self.decoder(x, enc_features, cond_image)

        # Step 5: Apply activation function based on the output channels
        if self.ndim == 1:
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output, dim=1)

        return output
    