"""
HyperNetwork implementation is based on

https://arxiv.org/abs/2106.06842
https://github.com/keynans/HypeRL
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block used for learnable task embeddings.
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class Head(nn.Module):
    """
    Hypernetwork head for generating weights of a single layer of an MLP.
    """
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super().__init__()

        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out

        self.W1 = nn.Linear(latent_dim, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(latent_dim, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)

        return w, b

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.b1.bias)


class Meta_Embadding(nn.Module):
    """
    Hypernetwork meta embedding.
    """
    def __init__(self, meta_dim, z_dim):
        super().__init__()

        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, z_dim // 4),
            ResBlock(z_dim // 4, z_dim // 4),
            ResBlock(z_dim // 4, z_dim // 4),

            nn.Linear(z_dim // 4, z_dim // 2),
            ResBlock(z_dim // 2, z_dim // 2),
            ResBlock(z_dim // 2, z_dim // 2),

            nn.Linear(z_dim // 2, z_dim),
            ResBlock(z_dim, z_dim),
            ResBlock(z_dim, z_dim),
        )

        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1. / (2. * math.sqrt(fan_in))
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork(nn.Module):
    """
    A hypernetwork that creates another neural network of
    base_v_input_dim -> base_v_output_dim using z_dim.
    """
    def __init__(self, meta_v_dim, z_dim, base_v_input_dim, base_v_output_dim,
                 dynamic_layer_dim, base_output_activation=None):
        super().__init__()

        self.base_output_activation = base_output_activation
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # main network
        self.layer1 = Head(z_dim, base_v_input_dim, dynamic_layer_dim, sttdev=0.05)
        self.last_layer = Head(z_dim, dynamic_layer_dim, base_v_output_dim, sttdev=0.008)

    def forward(self, meta_v, base_v):
        # produce dynamic weights
        z = self.hyper(meta_v)
        w1, b1 = self.layer1(z)
        w2, b2 = self.last_layer(z)

        # dynamic network pass
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) + b1)
        out = torch.bmm(w2, out) + b2
        if self.base_output_activation is not None:
            out = self.base_output_activation(out)

        batch_size = out.shape[0]
        return z, out.view(batch_size, -1)


class DoubleHeadedHyperNetwork(nn.Module):
    """
    A hypernetwork that creates two neural networks of
    base_v_input_dim[i] -> base_v_output_dim[i] using z_dim.
    """
    def __init__(self, meta_v_dim, z_dim, base_v_input_dim, base_v_output_dim,
                 dynamic_layer_dim, base_output_activation=None):
        super().__init__()
        assert isinstance(base_v_input_dim, list)
        assert isinstance(base_v_output_dim, list)
        assert isinstance(base_output_activation, list) or base_output_activation is None

        self.base_output_activation = base_output_activation
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # main networks
        self.layer1_1 = Head(z_dim, base_v_input_dim[0], dynamic_layer_dim, sttdev=0.05)
        self.last_layer_1 = Head(z_dim, dynamic_layer_dim, base_v_output_dim[0], sttdev=0.008)

        self.layer1_2 = Head(z_dim, base_v_input_dim[1], dynamic_layer_dim, sttdev=0.05)
        self.last_layer_2 = Head(z_dim, dynamic_layer_dim, base_v_output_dim[1], sttdev=0.008)

    def forward(self, meta_v, base_v_1, base_v_2):
        z = self.hyper(meta_v)
        out_1 = self.forward_net_1(z, base_v_1)
        out_2 = self.forward_net_2(z, base_v_2)
        return z, out_1, out_2

    def embed(self, meta_v):
        z = self.hyper(meta_v)
        return z

    def forward_net_1(self, z, base_v_1):
        # produce dynamic weights for network #1
        w1_1, b1_1 = self.layer1_1(z)
        w2_1, b2_1 = self.last_layer_1(z)

        # dynamic network 1 pass
        out_1 = F.relu(torch.bmm(w1_1, base_v_1.unsqueeze(2)) + b1_1)
        out_1 = torch.bmm(w2_1, out_1) + b2_1
        if self.base_output_activation[0] is not None:
            out_1 = self.base_output_activation[0](out_1)

        batch_size = out_1.shape[0]
        return out_1.view(batch_size, -1)

    def forward_net_2(self, z, base_v_2):
        # produce dynamic weights for network #2
        w1_2, b1_2 = self.layer1_2(z)
        w2_2, b2_2 = self.last_layer_2(z)

        # dynamic network 2 pass
        out_2 = F.relu(torch.bmm(w1_2, base_v_2.unsqueeze(2)) + b1_2)
        out_2 = torch.bmm(w2_2, out_2) + b2_2
        if self.base_output_activation[1] is not None:
            out_2 = self.base_output_activation[1](out_2)

        batch_size = out_2.shape[0]
        return out_2.view(batch_size, -1)
