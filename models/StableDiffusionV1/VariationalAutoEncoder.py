import torch
from torch import nn
import yaml

with open("Parameters.yaml", "r") as f:
    config = yaml.safe_load(f)["VAE"]

class ResNet(nn.Module):
    """
    """
    def __init__(self, in_channels:int, out_channels:int, num_groups:int):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6
        )
        self.groupnorm2 = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels, eps=1e-6
        )
        self.silu1 = nn.SiLU()
        self.silu2 = nn.SiLU()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.residual_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1,1), padding=0, stride=1
        )

    def forward(self, x:torch.Tensor):
        residual = x
        x = self.groupnorm1(x)
        x = self.silu1(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.silu2(x)
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x


class SelfAttention(nn.Module):
    """
    """
    def __init__(self, num_groups:int, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.channels = channels
        self.query_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.key_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.value_conv = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )
        self.projection = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(1,1), stride=1,
            padding=0
        )

    def forward(self, x:torch.Tensor):
        residual = x
        batch, _, height, width = x.shape
        x = self.groupnorm(x)
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        query = query.permute(0,2,3,1).flatten(start_dim=1,end_dim=2)
        key = key.flatten(start_dim=2,end_dim=3)
        value = value.flatten(start_dim=2,end_dim=3)

        attn_scores = query @ key.transpose(-1,-2)
        attn_scores = attn_scores / (self.channels ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = attn_scores @ value

        attn_scores = attn_scores.permute(0,2,1).reshape(batch, self.channels, height, width)
        return residual + self.projection(attn_scores)


class Downsample(nn.Module):
    """
    """
    def __init__(self, input_channels:int, output_channels:int):
        super().__init__()
        self.downsampling_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=2,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.downsampling_conv(x)
        return x

class Upsample(nn.Module):
    """
    """
    def __init__(self, input_channels:int, output_channels:int, scale_factor:int=config["upsample_scale_factor"]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.upsampling_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.upsample(x)
        x = self.upsampling_conv(x)
        return x

class Encoder(nn.Module):
    """
    """
    def __init__(self, embed_dim:int, input_channels:int, output_channels:int, rgb_channels:int, num_groups:int):
        super().__init__()
        self.input_channels = input_channels
        self.input_channelsx2 = input_channels * 2
        self.input_channelsx4 = input_channels * 4
        self.input_conv = nn.Conv2d(
            in_channels=rgb_channels,
            out_channels=input_channels,
            kernel_size=(3,3),
            stride=1,
            padding=1
        )
        self.down_block1 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                Downsample(input_channels=self.input_channels, output_channels=self.input_channelsx2)
            ]
        )
        self.down_block2 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx2, out_channels=self.input_channelsx2),
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx2, out_channels=self.input_channelsx2),
                Downsample(input_channels=self.input_channelsx2, output_channels=self.input_channelsx4)
            ]
        )
        self.down_block3 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4),
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4),
                Downsample(input_channels=self.input_channelsx4, output_channels=self.input_channelsx4)
            ]
        )
        self.down_block4 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4),
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4)
            ]
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4),
                SelfAttention(num_groups=num_groups, channels=self.input_channelsx4),
                ResNet(num_groups=num_groups, in_channels=self.input_channelsx4, out_channels=self.input_channelsx4)
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=self.input_channelsx4, eps=1e-6)
        self.silu = nn.SiLU(inplace=True)
        self.proj_conv = nn.Conv2d(
            in_channels=self.input_channelsx4, out_channels=output_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.quant_conv = nn.Conv2d(
            in_channels=output_channels, out_channels=2*embed_dim,
            kernel_size=(1,1), stride=1, padding=0
        )
        self.latent_distribution = LatentDistribution()
    def forward(self, x:torch.Tensor):
        x = self.input_conv(x)
        for block1 in self.down_block1:
            x = block1(x)
        for block2 in self.down_block2:
            x = block2(x)
        for block3 in self.down_block3:
            x = block3(x)
        for block4 in self.down_block4:
            x = block4(x)
        for block5 in self.middle_block:
            x = block5(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.proj_conv(x)
        x = self.quant_conv(x)
        z = self.latent_distribution(x)
        return z


class Decoder(nn.Module):
    """
    """
    def __init__(self, embed_dim:int, rgb_channels:int, latent_channels:int, input_channels:int, num_groups:int):
        super().__init__()
        self.input_channels = input_channels
        self.input_channels_2 = input_channels // 2
        self.input_channels_4 = input_channels // 4
        self.post_quant_conv = nn.Conv2d(
            in_channels=embed_dim, out_channels=latent_channels,
            kernel_size=(1,1), stride=1, padding=0
        )
        self.proj_conv = nn.Conv2d(
            in_channels=latent_channels, out_channels=input_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
        self.middle_block = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                SelfAttention(num_groups=num_groups, channels=input_channels),
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels)
            ]
        )
        self.up_block1 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                ResNet(num_groups=num_groups, in_channels=input_channels, out_channels=input_channels),
                Upsample(input_channels=input_channels, output_channels=self.input_channels_2)
            ]
        )
        self.up_block2 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_2, out_channels=self.input_channels_2),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_2, out_channels=self.input_channels_2),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_2, out_channels=self.input_channels_2),
                Upsample(input_channels=self.input_channels_2, output_channels=self.input_channels_4)            
            ]
        )
        self.up_block3 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4),
                Upsample(input_channels=self.input_channels_4, output_channels=self.input_channels_4)
            ]
        )
        self.up_block4 = nn.ModuleList(
            [
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4),
                ResNet(num_groups=num_groups, in_channels=self.input_channels_4, out_channels=self.input_channels_4)
            ]
        )
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=self.input_channels_4, eps=1e-6)
        self.silu = nn.SiLU(inplace=True)
        self.output_conv = nn.Conv2d(
            in_channels=self.input_channels_4, out_channels=rgb_channels,
            kernel_size=(3,3), stride=1, padding=1
        )
    def forward(self, x:torch.Tensor):
        x = self.post_quant_conv(x)
        x = self.proj_conv(x)
        for block1 in self.middle_block:
            x = block1(x)
        for block2 in self.up_block1:
            x = block2(x)
        for block3 in self.up_block2:
            x = block3(x)
        for block4 in self.up_block3:
            x = block4(x)
        for block5 in self.up_block4:
            x = block5(x)
        x = self.groupnorm(x)
        x = self.silu(x)
        x = self.output_conv(x)
        return x

class LatentDistribution(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor):
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        mean, log_variance = x.chunk(chunks=2, dim=1)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        std = torch.exp(log_variance * 0.5)
        z = (noise * std) + mean
        return z

class VAE(nn.Module):
    """
    """
    def __init__(self, encode:bool, decode:bool):
        super().__init__()
        self.encode = encode
        self.decode = decode
        if self.encode:
            self.decode = False
        if self.decode:
            self.encode = False
        
        embed_dim = config["embed_dim"]
        num_groups = config["num_groups"]
        latent_channels = config["latent_channels"]
        rgb_channels = config["rgb_channels"]
        encoder_input_channels = config["encoder_input_channels"]
        encoder_output_channels = config["encoder_output_channels"]
        decoder_input_channels = config["decoder_input_channels"]
        decoder_output_channels = config["decoder_output_channels"]

        self.encoder = Encoder(embed_dim=embed_dim,input_channels=encoder_input_channels,
                               output_channels=encoder_output_channels, rgb_channels=rgb_channels, num_groups=num_groups)
        self.decoder = Decoder(embed_dim=embed_dim,rgb_channels=decoder_output_channels,
                               latent_channels=latent_channels, input_channels=decoder_input_channels, num_groups=num_groups)

    def forward(self, x:torch.Tensor):
        if self.encode: x = self.encoder(x)
        elif self.decode: x = self.decoder(x)
        return x

    def load_pretrained(self):
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.encoder.input_conv.weight.copy_(
            vae.encoder.conv_in.weight
        )
        self.encoder.input_conv.bias.copy_(
            vae.encoder.conv_in.bias
        )
        



