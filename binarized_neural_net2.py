###############################################################################
#
# Provides the required elements to implement a binary convolutional network in
# PyTorch.
#
# This file contains the following elements are implemented:
# * BinaryLinear
# * BinaryConv2d
# * sign function with straight-through estimator gradient
# * Binary optimization algorithm
#
# Inspiration taken from:
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
#
# Author(s): Nik Vaessen
###############################################################################

from typing import TypeVar, Union, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim import Adam

################################################################################

# taken from https://github.com/pytorch/pytorch/blob/bfeff1eb8f90aa1ff7e4f6bafe9945ad409e2d97/torch/nn/common_types.pyi

T = TypeVar("T")
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]

################################################################################
# Quantizers


class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        output = inp.sign()
        # output = inp.new(inp.size())
        # output[inp >= 0] = 1
        # output[inp < 0] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        clipped = inp.abs() <= Binarize.clip_value

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output


binarize = Binarize.apply

################################################################################
# binary torch layers


class BinaryLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)
        out = f.linear(inp, weight, bias)
        return out


class BinaryConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride=1,
        padding=1,
        bias=False,
        keep_latent_weight=False,
        binarize_input=False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

        self.keep_latent_weight = keep_latent_weight
        self.binarize_input = binarize_input

        if not self.keep_latent_weight:
            with torch.no_grad():
                self.weight.data.sign_()
                self.bias.data.sign_() if self.bias is not None else None

    def forward(self, inp: Tensor) -> Tensor:
        if self.keep_latent_weight:
            weight = binarize(self.weight)
        else:
            weight = self.weight

        bias = self.bias if self.bias is None else binarize(self.bias)

        if self.binarize_input:
            inp = binarize(inp)

        return f.conv2d(
            inp, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


# Define the NN architecture
class BinaryNet2(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers=2, num_classes=2):
        super(BinaryNet2, self).__init__()
        self.fc_list = nn.ModuleList()
        self.num_classes = num_classes
        if num_classes == 2:
            output_net_size = 1
        else:
            output_net_size = num_classes
        self.input_size = input_size
        input_layer_size = input_size
        output_layer_size = hidden_dim

        for i in range(n_layers):
            if i == n_layers - 1:
                output_layer_size = output_net_size
            self.fc_list.append(BinaryLinear(input_layer_size, output_layer_size, bias=False, binarize_input=True))
            input_layer_size = hidden_dim

        self.dropout = nn.Dropout(0.2)
        self.activation = binarize

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.input_size)
        # add hidden layer, with relu activation function
        for fc_layer in self.fc_list:
            # TODO: Add dropout to middle layers
            x = self.activation(fc_layer(x))
        if self.num_classes == 2:
            x += 1
            x /= 2
            x.squeeze_()
        return x
