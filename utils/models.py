import torch
import torch.nn as nn
import copy

# ── Fully Connected Neural Network ──────────────────────────────────────────────────────────
class FCNN(nn.Module):
    """
    Fully-connected neural network for 1-D regression.
 
    Args:
        h_n  : number of hidden layers
        n_n  : number of nodes in each hidden layer
        func : activation function applied after every hidden layer
    """
    def __init__(self, h_n: int, n_n: int, func: nn.Module):
        super().__init__()
        self.h_n  = h_n
        self.n_n  = n_n
        self.func = func
 
        layers = [nn.Linear(1, n_n), copy.deepcopy(func)]
        for _ in range(h_n - 1):
            layers += [nn.Linear(n_n, n_n), copy.deepcopy(func)]
        layers.append(nn.Linear(n_n, 1))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Convolutional Neural Network ───────────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    1-D convolutional neural network for sequence regression.

    The input (N, 1) is treated as a single-channel sequence of length N.
    h_n conv layers (each with n_n filters, kernel 3, same padding) are
    followed by a pointwise Conv1d head that maps n_n -> 1 at every position.

    Args:
        h_n  : number of convolutional hidden layers
        n_n  : number of filters (channels) in every conv layer
        func : activation function applied after every conv layer
    """
    def __init__(self, h_n: int, n_n: int, kernel_size: int, padding: int, stride: int, func: nn.Module):
        super().__init__()
        self.h_n  = h_n
        self.n_n  = n_n
        self.func = func
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        conv_layers: list = []
        in_ch = 1
        for _ in range(h_n):
            conv_layers += [
                nn.Conv1d(in_ch, n_n, kernel_size=kernel_size, padding=padding, stride=stride),
                copy.deepcopy(func),
            ]
            in_ch = n_n
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, 1)
        # Conv1d expects (batch, channels, length) -> reshape to (1, 1, N)
        out = x.T.unsqueeze(0)   # (1, 1, N)
        out = self.conv(out)     # (1, n_n, N)
        out = self.head(out)     # (1, 1, N)
        return out.squeeze(0).T  # (N, 1)