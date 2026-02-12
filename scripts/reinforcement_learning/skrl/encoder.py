from typing import Dict, List, Optional

import torch
from gymnasium import spaces
from torch import Tensor, nn
from learning_utils import *




# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(nn.Module):
    """
    The base class for all encoders. Inherits from torch.nn.Module.
    """
    def __init__(self, cfg: Dict):
        # Call the __init__ of the parent class (nn.Module)
        super().__init__()
        # Store the configuration dictionary
        self.cfg = cfg

    def get_out_size(self) -> int:
        """
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError()

class ResBlock(nn.Module):
    def __init__(self, cfg : Dict, input_ch, output_ch):
        super().__init__()
        activation = get_activation(cfg['nonlinearity'])
        layers = [
            activation,
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            activation,
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out

class ResBlock3D(nn.Module):
    def __init__(self, cfg: Dict, input_ch: int, output_ch: int):
        super().__init__()
        activation = get_activation(cfg['nonlinearity'])
        layers = [
            activation,
            nn.Conv3d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            activation,
            nn.Conv3d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    def __init__(self, cfg: Dict, obs_space: spaces.Box):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        print(f"2D ResNet Encoder: Num input channels: {input_ch}")

        # configuration from the IMPALA paper
        resnet_conf = [[16, 2], [32, 2], [32, 2]]

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = get_activation(cfg['nonlinearity'])
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = get_conv_output_size(self.conv_head, obs_space.shape)
        print(f"Convolutional layer output size: {self.conv_head_out_size}")

        self.mlp_layers = create_mlp(cfg['encoder_conv_mlp_layers'], self.conv_head_out_size, activation)

        # should we do torch.jit here?
        self.encoder_out_size = cfg['encoder_conv_mlp_layers'][-1]

        #self.encoder_out_size = get_conv_output_size(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.mlp_layers[0].in_features)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class Resnet3DEncoder(Encoder):
    def __init__(self, cfg, obs_space, type):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        print("Num input channels: %d", input_ch)

            # configuration from the IMPALA paper
        # configuration from the IMPALA paper
        # resnet_conf = [[8, 2], [16, 2], [16, 2]]
        resnet_conf = [[16, 2], [32, 2], [32, 2]]
       
        curr_input_channels = input_ch
        layers = []

        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv3d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock3D(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = get_activation(cfg['nonlinearity'])
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)

        conv_out_size = get_conv_output_size(self.conv_head, obs_space.shape)
        print(f"3D ResNet conv head output size: {conv_out_size}")

        # should we do torch.jit here?
        mlp_layers_cfg = cfg['encoder_conv_map_occupancy_mlp_layers']

        self.mlp_layers = create_mlp(mlp_layers_cfg, conv_out_size, activation)

        self.encoder_out_size = mlp_layers_cfg[-1]

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.mlp_layers[0].in_features)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


# def make_img_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:

#     if cfg['encoder_conv_architecture'] == "resnet_impala":
#         return ResnetEncoder(cfg, obs_space)
#     else:
#         raise NotImplementedError(f"Unknown convolutional architecture {cfg['encoder_conv_architecture']}")

# def make_map_encoder(cfg: Config, obs_space: ObsSpace, type) -> Encoder:
#     """Make (most likely convolutional) encoder for 3Dmap-based observations."""

#     if cfg['encoder_conv_map_occupancy_architecture'] == "resnet":
#         return Resnet3DEncoder(cfg, obs_space)
#     else:
#         raise NotImplementedError(f"Unknown convolutional architecture {cfg['encoder_conv_map_occupancy_architecture']}")