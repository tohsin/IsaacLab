import torch
from torch import nn, Tensor
from gymnasium import spaces
from typing import List, Dict
import numpy as np

def get_activation(activation_name: str) -> nn.Module:
    """Returns a PyTorch activation function."""
    if activation_name.lower() in ["relu", "rectifier"]:
        return nn.ReLU()
    elif activation_name.lower() == "elu":
        return nn.ELU()
    elif activation_name.lower() == "tanh":
        return nn.Tanh()
    elif activation_name.lower() in ["silu", "swish"]:
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")
    
def create_mlp(layer_sizes: List[int], input_size: int, activation_func: nn.Module) -> nn.Sequential:
    """Creates a Multi-Layer Perceptron (MLP) as an nn.Sequential module."""
    layers = []
    in_size = input_size
    for out_size in layer_sizes:
        layers.append(nn.Linear(in_size, out_size))
        layers.append(activation_func)
        in_size = out_size
    return nn.Sequential(*layers)

def get_conv_output_size(conv_module: nn.Module, input_shape: tuple) -> int:
    """Calculates the flattened output size of a convolutional module."""
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = conv_module(dummy_input)
        return int(np.prod(output.shape))