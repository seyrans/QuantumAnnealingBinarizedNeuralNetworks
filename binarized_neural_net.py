from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
from utils import to_boolean
import math


# BNN code adapted from:
# github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch
class BinarizeF(Function):

    @staticmethod
    def forward(cxt, input_):
        output = input_.new(input_.size())
        output[input_ >= 0] = 1
        output[input_ < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# aliases
binarize_to_spin = BinarizeF.apply


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input_):
        output = self.hardtanh(input_)
        output = binarize_to_spin(output)
        return output


class BinaryLinear(nn.Linear):

    def forward(self, input_):
        binary_weight = binarize_to_spin(self.weight)
        if self.bias is None:
            return F.linear(input_, binary_weight)
        else:
            return F.linear(input_, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


class BinaryConv2d(nn.Conv2d):

    def forward(self, input_):
        bw = binarize_to_spin(self.weight)
        return F.conv2d(input_, bw, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


def valid(size):
    base2_log = math.log2(size + 1)
    if int(base2_log) != base2_log:
        raise ValueError('Input size must be in a form of 2**n+1.')


# Define the NN architecture
class BinaryNet(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers=2, num_classes=2):
        valid(hidden_dim)
        super(BinaryNet, self).__init__()
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
            self.fc_list.append(BinaryLinear(input_layer_size, output_layer_size, bias=False))
            input_layer_size = hidden_dim

        self.dropout = nn.Dropout(0.2)
        self.activation = binarize_to_spin

    def forward(self, x, keep_intermediate=False):
        # flatten image input
        x = x.view(-1, self.input_size)
        valid(x.shape[-1])
        # add hidden layer, with relu activation function
        if keep_intermediate:
            nn_lin_outs = []
            nn_lay_outs = []
            nn_weights = []
        for fc_layer in self.fc_list:
            # TODO: Add dropout to middle layers
            lin_out = fc_layer(x)
            x = self.activation(lin_out)
            if keep_intermediate:
                nn_lin_outs.append(lin_out)
                nn_lay_outs.append(x)
                nn_weights.append(binarize_to_spin(fc_layer.weight))

        if self.num_classes == 2:   # converts the result from spin to boolean
            x = to_boolean(x)
            x = x.squeeze()
        if keep_intermediate:
            x.nn_lin_outs = nn_lin_outs
            x.nn_lay_outs = nn_lay_outs
            x.nn_weights = nn_weights
        return x
