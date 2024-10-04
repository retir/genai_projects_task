import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry


diffusion_models_registry = ClassRegistry()


class VerySimpleUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(-0.2) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


@diffusion_models_registry.add_to_registry(name="base_diffusion")
class VerySimpleUnet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_steps = model_config["num_steps"]
        self.base_hidden_dim = model_config["base_hidden_dim"]
        self.blocks_num = model_config["blocks_num"]

        self.input_conv = nn.Conv2d(3, self.base_hidden_dim, kernel_size=3, stride=1, padding=1)
        self.time_embed = nn.Embedding(self.num_steps, self.base_hidden_dim)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        for i in range(self.blocks_num):
            lower_dim = self.base_hidden_dim * 2**i
            higher_dim = lower_dim * 2

            self.down_blocks.append(VerySimpleUnetBlock(lower_dim, higher_dim))
            self.up_blocks.append(VerySimpleUnetBlock(higher_dim, lower_dim))


    def forward(self, x, t):
        t = self.time_embed(t)
        t = t[..., None, None]

        x = self.input_conv(x)
        x = x + t

        for block in self.down_blocks:
            x = block(x)
            x = F.interpolate(x, x.size(-1) // 2)

        for block in reversed(self.up_blocks):
            x = block(x)
            x = F.interpolate(x, x.size(-1) * 2)
            
        return x