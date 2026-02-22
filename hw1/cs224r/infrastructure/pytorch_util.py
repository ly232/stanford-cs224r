"""
TO EDIT: Utilities for handling PyTorch

Functions to edit:
    1. build_mlp (line 26)
"""

from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

device = None


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Module:
    """
    Builds a feedforward neural network

    Example mental model:
    - imagine a robot with 10 joints.
    - state space is position, velocity and acceleration at each joint.
    - input_size = 30 (10 positions, 10 velocities, 10 accelerations)
    - output_size = 10 (mean action for each joint)

    Arguments:
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer

        input_size: size of the input layer, i.e. dimensionality of the state
          space.
        output_size: size of the output layer, indicating the number of action
          dimensions (e.g. each joint has its own mean, so 10 joints would have
          output_size = 10)
        output_activation: activation of the output layer

    Returns:
        MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    # TODO[DONE]: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    layers = []
    layers.append(nn.Linear(input_size, size))
    layers.append(activation)
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(size, size))
        layers.append(activation)
    layers.append(nn.Linear(size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("PyTorch detects an Apple GPU: running on MPS")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to("cpu").detach().numpy()
