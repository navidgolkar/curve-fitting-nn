import torch
import torch.nn as nn
import copy
from typing import List


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
        layers = []

        if h_n == 0:
            layers.append(nn.Linear(1, 1))
        else:
            layers.extend([nn.Linear(1, n_n), copy.deepcopy(func)])
            for _ in range(h_n - 1):
                layers.extend([nn.Linear(n_n, n_n), copy.deepcopy(func)])
            layers.append(nn.Linear(n_n, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Convolutional Neural Network ───────────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    1-D convolutional neural network for sequence regression.

    Args:
        h_n  : number of convolutional hidden layers
        n_n  : number of filters (channels) in every conv layer
        func : activation function applied after every conv layer
    """
    def __init__(self, h_n: int, n_n: int, kernel_size: int, padding: int,
                 stride: int, func: nn.Module):
        super().__init__()
        self.h_n         = h_n
        self.n_n         = n_n
        self.func        = func
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride

        conv_layers: list = []
        conv_layers.extend([
            nn.Conv1d(1, n_n, kernel_size=1),
            copy.deepcopy(func),
        ])
        for _ in range(h_n-1):
            in_ch = n_n
            conv_layers.extend([
                nn.Conv1d(in_ch, n_n, kernel_size=kernel_size,
                          padding=padding, stride=stride),
                copy.deepcopy(func),
            ])
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x   = x.permute(0, 2, 1)       # (B, 1, N)
        out = self.conv(x)
        out = self.head(out)
        out = out.permute(0, 2, 1)      # (B, N, 1)
        return out.squeeze(0)


# ── Shared residual layer building blocks ──────────────────────────────────────────────────

class _DenseResLayer(nn.Module):
    """
    One layer of a DenseResNet (layer index i ≥ 1, 0-indexed).

    Receives the concatenation of ALL previous hidden states h_0 … h_{i-1},
    applies a linear transform + activation, then adds h_{i-1} as the
    identity residual:

        h_i = func(W · [h_0 ‖ … ‖ h_{i-1}]) + h_{i-1}

    Args:
        in_size  : total width of the concatenated input (= i × n_n, i ≥ 2).
        out_size : output width = n_n.
        func     : activation applied after the linear transform.
    """
    def __init__(self, in_size: int, out_size: int, func: nn.Module):
        super().__init__()
        self.out_size = out_size
        self.linear   = nn.Linear(in_size, out_size)
        self.func     = copy.deepcopy(func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, in_size) — concat of all previous layer outputs
        transformed = self.func(self.linear(x))   # (N, out_size)
        residual    = x[..., -self.out_size:]      # h_{i-1}: last out_size cols
        return transformed + residual              # (N, out_size)


class _ConvResLayer(nn.Module):
    """
    One convolutional residual layer of a ConvResNet (layer index i ≥ 1).

    Applies conv + activation to its input, then adds the input as the
    identity residual.  Extra skip connections from earlier layers are
    summed into the input *before* this module is called.

        h_i = func(Conv(x)) + x      where x = h_{i-1} + Σ skip sources

    Args:
        in_n         : input number of channels / filters.
        out_n         : output number of channels / filters.
        kernel_size : Conv1d kernel size.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
        func        : activation applied after the convolution.
    """
    def __init__(self, in_n: int, out_n: int, kernel_size: int, padding: int,
                 stride: int, func: nn.Module):
        super().__init__()
        self.conv = nn.Conv1d(in_n, out_n, kernel_size,
                              padding=padding, stride=stride)
        self.func = copy.deepcopy(func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(self.conv(x)) + x         # (n_n, N_seq)


# ── Dense Residual Network ────────────────────────────────────────────────────────────────
class DenseResNet(nn.Module):
    """
    Dense Residual Network for 1-D regression.

    Layer 0  — first hidden layer: Linear(1 → n_n) + func.
               No residual (there is no preceding hidden state).
    Layer i  — (i = 1 … h_n-1): _DenseResLayer.
               Input = concat(h_0, … h_{i-1}), width = i * n_n.
               Output = func(W · input) + h_{i-1}, width = n_n.
    Head     — Linear(n_n → 1).

    Skip-connection rule (0-indexed over all h_n hidden layers):
        Layer i sends its output as a long-range skip to every j where
        j > i + 1.  Layer j = i+1 already receives h_i as the identity
        residual inside _DenseResLayer, so it is not a skip target.

    Attributes:
        _skip_targets : list[list[int]]
            _skip_targets[i] = [j  for j in range(h_n) if j > i+1]
        _skip_sources : list[list[int]]
            _skip_sources[j] = [i  for i in range(h_n) if i < j-1]

    Args:
        h_n  : total number of hidden layers (≥ 1).
        n_n  : nodes per hidden layer.
        func : activation function.
    """
    def __init__(self, h_n: int, n_n: int, func: nn.Module):
        super().__init__()
        self.h_n = h_n
        self.n_n = n_n

        # ── Skip-connection maps ─────────────────────────────────────────
        self._skip_targets: list[list[int]] = [
            [j for j in range(h_n) if j > i + 1]
            for i in range(h_n)
        ]
        self._skip_sources: list[list[int]] = [
            [i for i in range(h_n) if i < j - 1]
            for j in range(h_n)
        ]

        # ── Build layers ─────────────────────────────────────────────────
        # Layer 0: plain Linear(1→n_n) + act, no residual.
        # Layer i (i≥1): _DenseResLayer with in_size = i*n_n.
        layer_list: list[nn.Module] = [
            nn.Sequential(nn.Linear(1, n_n), copy.deepcopy(func))
        ]
        for i in range(1, h_n):
            layer_list.append(_DenseResLayer(i * n_n, n_n, func))
        self.layers = nn.ModuleList(layer_list)

        self.head = nn.Linear(n_n, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 0: (N,1) → (N, n_n)
        h       = self.layers[0](x)
        outputs = [h]

        # Layers 1 … h_n-1
        for i in range(1, self.h_n):
            # concat(h_0, …, h_{i-1})  →  (N, i*n_n)
            cat = torch.cat(outputs, dim=-1)
            h   = self.layers[i](cat)           # (N, n_n)
            outputs.append(h)

        return self.head(outputs[-1])           # (N, 1)


# ── Convolutional Residual Network ─────────────────────────────────────────────────────────
class ConvResNet(nn.Module):
    """
    Convolutional Residual Network for 1-D regression.

    There are exactly h_n hidden layers and no separate input_proj layer.

    Layer 0  — first hidden layer: Linear(1 → n_n) + func.
               Expands the scalar input to n_n channels; no conv, no residual.
    Layer i  — (i = 1 … h_n-1): _ConvResLayer.
               Input = h_{i-1} + Σ_{k ∈ _skip_sources[i]} h_k.
               Output = func(Conv(input)) + input.
    Head     — Conv1d(n_n → 1, k=1).

    Skip-connection rule (0-indexed over all h_n hidden layers):
        Layer i sends an extra skip to every j where i+1 < j ≤ i+connect+1,
        BUT ONLY when i + connect + 2 ≤ h_n.
        When that condition does not hold, layer i sends no extra skips.

    Attributes:
        _skip_targets : list[list[int]]
            _skip_targets[i] = target layer indices j.
        _skip_sources : list[list[int]]
            _skip_sources[j] = source layer indices i (inverted from targets).

    Args:
        h_n         : total number of hidden layers (≥ 1).
        n_n         : channels / filters per layer.
        kernel_size : Conv1d kernel size inside each _ConvResLayer.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
        func        : activation function.
        connect     : extra-skip window size.  connect=0 → pure ResNet.
    """
    def __init__(self, h_n: int, n_n: int, kernel_size: int, padding: int,
                 stride: int, func: nn.Module, connect: int = 1):
        super().__init__()
        self.h_n         = h_n
        self.n_n         = n_n
        self.connect     = connect
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride

        # ── Skip-connection maps ─────────────────────────────────────────
        self._skip_targets: list[list[int]] = []
        for i in range(h_n):
            if i + connect + 2 <= h_n:
                targets = [j for j in range(h_n)
                           if i + 1 < j <= i + connect + 1]
            else:
                targets = []
            self._skip_targets.append(targets)

        self._skip_sources: list[list[int]] = [[] for _ in range(h_n)]
        for i, targets in enumerate(self._skip_targets):
            for j in targets:
                self._skip_sources[j].append(i)

        # ── Build layers ─────────────────────────────────────────────────
        # Layer 0: Linear(1→n_n) + act (no conv, no residual).
        # Layers 1…h_n-1: _ConvResLayer.
        layer_list: list[nn.Module] = [
            nn.Sequential(nn.Linear(1, n_n), copy.deepcopy(func))
        ]
        for _ in range(1, h_n):
            layer_list.append(_ConvResLayer(n_n, n_n, kernel_size, padding, stride, func))
        self.layers = nn.ModuleList(layer_list)

        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 0: (N,1) → (N, n_n) → (n_n, N) for Conv1d
        h       = self.layers[0](x).T
        outputs = [h]                     # outputs[k] = h_k  shape (n_n, N)

        # Layers 1 … h_n-1
        for idx in range(1, self.h_n):
            layer_input = outputs[idx - 1]
            for src_idx in self._skip_sources[idx]:
                layer_input = layer_input + outputs[src_idx]
            h = self.layers[idx](layer_input)   # (n_n, N)
            outputs.append(h)

        out = self.head(outputs[-1].unsqueeze(0))   # (1, 1, N)
        return out.squeeze(0).T                     # (N, 1)