import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_dims=[256, 256, 256],
                 activation="relu",
                 norm="none",
                 requires_grad=True,
                 init_weights=False,
                 **kwargs):
        super(MLP, self).__init__()

        # local vriables
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dims = hidden_dims

        layers = []
        for l in range(len(hidden_dims)):
            if l == 0:
                layers.append(nn.Linear(input_size, hidden_dims[l]))
            else:
                layers.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))
            layers.append(get_activation(activation))

        layers.append(nn.Linear(hidden_dims[-1], output_size))

        if norm is not None and norm != "none":
            layers.append(get_norm(norm))

        self.model = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = requires_grad

        if init_weights:
            self.init_weights()

    def forward(self, x):
        return self.model(x)

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)


def get_activation(act_name):
    """Get activation function from name.

    Args:
        act_name (str): activation function name.

    Returns:
        nn.functional: pytorch activation function.
    """
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def get_norm(norm_name):
    if norm_name == "batch":
        return nn.BatchNorm1d()
    elif norm_name == "layer":
        return nn.LayerNorm()
    elif norm_name == "instance":
        return nn.InstanceNorm1d()
    elif norm_name == "softmax":
        return nn.Softmax(dim=-1)
    elif norm_name == "none":
        return None
    else:
        print("MLP: invalid normalization function!")
        return None
