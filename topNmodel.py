import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        output_dim = input_dim
        self.layer1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.ReLU(out)
        out = self.layer2(out)
        out += x
        out = self.ReLU(out)
        return out

class Dis_MyModel_GAN_8(nn.Module):

    def __init__(self, n_items, hidden_dim=10000, output_activation='sigmoid'):
        super(Dis_MyModel_GAN_8, self).__init__()
        self.hidden_layer1 = nn.Linear(n_items, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim - (hidden_dim - n_items) // 2)
        self.output_layer = nn.Linear(hidden_dim - (hidden_dim - n_items) // 2, n_items)
        self.mlp1 = nn.Linear(hidden_dim, n_items)

        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()
        if output_activation == 'sigmoid':
            print('output_layer activation is sigmoid')
            self.output_activation = self.sigmoid
        elif output_activation == 'ReLU':
            print('output_layer activation is ReLU')
            self.output_activation = self.ReLU
        else:
            print('output_layer activation is LeakyReLU')
            self.output_activation = self.LeakyReLU

    def forward(self, x):
        # input, pad = x  # (BS, seq_len)
        # batch_size, seq_len = input.shape
        input = x

        layer1_output = self.hidden_layer1(input)
        layer1_output_acti = self.Tanh(layer1_output)
        # # layer1_output_result = input + layer1_output_acti
        #
        # layer2_output = self.hidden_layer2(layer1_output_acti)
        # layer2_output_acti = self.LeakyReLU(layer2_output)
        # # layer2_output_result = layer2_output + layer2_output_acti
        #
        # o_layer_output = self.output_layer(layer2_output_acti)

        # return self.output_activation(o_layer_output)
        return self.output_activation(self.mlp1(layer1_output_acti))


class Gen_MyModel_GAN_8(nn.Module):

    def __init__(self, n_items, hidden_dim):
        super(Gen_MyModel_GAN_8, self).__init__()
        self.hidden_layer = nn.Linear(n_items, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, n_items)

        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        hidden_output = self.hidden_layer(x)
        hidden_output_acti = self.ReLU(hidden_output)
        hidden_output1 = self.hidden_layer1(hidden_output_acti)
        hidden_output1_acti = self.ReLU(hidden_output1)
        output_layer_output = self.output_layer(hidden_output1_acti)
        return self.sigmoid(output_layer_output)
        # return output_layer_output