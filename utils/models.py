import torch
import torch.nn as nn
import numpy as np
import copy

# ── Connectivity helpers ──────────────────────────────────────────────────────
def _full_connections(n_n: int) -> np.ndarray:
    """
    All-True (n_n × n_n) matrix for a Linear(n_n → n_n) layer.
    Entry [out, in] = True  ↔  out-node receives from in-node.
    Every output reads every input in a dense layer.
    """
    return np.ones((n_n, n_n), dtype=bool)
 
 
def _conv_connections(n_n: int, kernel_size: int,
                      padding: int, stride: int) -> np.ndarray:
    """
    Sparse (n_n × n_n) connectivity matrix for Conv1d(n_n, n_n, ...).
    Entry [out_filter, in_filter] = True  ↔  out_filter's receptive field
    overlaps with in_filter.
 
    Receptive-field window of output filter j over input filters:
        [ j*stride - padding,  j*stride - padding + kernel_size - 1 ]  ∩  [0, n_n)
    """
    conn = np.zeros((n_n, n_n), dtype=bool)
    for j in range(n_n):
        i_min = j * stride - padding
        i_max = i_min + kernel_size - 1
        for i in range(n_n):
            if i_min <= i <= i_max:
                conn[j, i] = True
    return conn

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
        
        # ── Connectivity ──────────────────────────────────────────────
        # Shape: (h_n, n_n, n_n).  _connections[i, out, in] = True when
        # out-node receives from in-node at the i-th hidden→hidden transition.
        # Every layer is Linear, so every entry is True.
        self._connections: np.ndarray = np.stack([_full_connections(n_n) for _ in range(max(h_n, 1))], axis=0)
        
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
    def __init__(self, h_n: int, n_n: int, kernel_size: int, padding: int, stride: int, func: nn.Module):
        super().__init__()
        self.h_n = h_n
        self.n_n = n_n
        self.func = func
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # ── Connectivity ──────────────────────────────────────────────
        # Shape: (h_n, n_n, n_n).
        # Layer 0: Conv1d(1, n_n, k=1) — single input channel fans out to
        #          all n_n filters equally → full.
        # Layers 1+: Conv1d(n_n, n_n, k, p, s) → sparse banded matrix.
        conv_conn = _conv_connections(n_n, kernel_size, padding, stride)
        self._connections: np.ndarray = np.stack([_full_connections(n_n)] + [conv_conn] * max(h_n - 1, 0), axis=0)
        
        conv_layers: list = []
        conv_layers.extend([nn.Conv1d(1, n_n, kernel_size=1), copy.deepcopy(func)])
        for _ in range(h_n-1):
            conv_layers.extend([nn.Conv1d(n_n, n_n, kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(func)])
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instead of permute you can Transpose (x.T) but to be consistent with ConvResNet which needs permute I used permute here as well
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)       # (B, 1, N)
        out = self.conv(x)
        out = self.head(out)
        out = out.permute(0, 2, 1)      # (B, N, 1)
        return out.squeeze(0)

# ── Dense Residual Network ────────────────────────────────────────────────────────────────
class DenseResNet(nn.Module):
    """
    Dense Residual Network for 1-D regression.

    FCNN + Skip(residual)-connections

    Skip-connection rule (0-indexed over all h_n hidden layers):
        Layer i sends its output as a long-range skip to every j where
        j > i + 1.

    Attributes:
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
        
        # ── Connectivity ──────────────────────────────────────────────
        # Shape: (h_n, n_n, n_n).  All direct transitions are Linear → full.
        # (_connections covers direct edges only; skip edges are in _skip_sources.)
        self._connections: np.ndarray = np.stack([_full_connections(n_n) for _ in range(max(h_n, 1))], axis=0)
        # ── Skip-connection maps ─────────────────────────────────────────
        self._skip_sources: list[list[int]] = [[i for i in range(h_n) if i < j - 1] for j in range(h_n)]

        # ── Build layers ─────────────────────────────────────────────────
        # Layer 0: plain Linear(1→n_n) + act, no residual.
        # Layer i (i≥1): Linear(n_n→n_n) + act.
        layer_list: list[nn.Module] = [nn.Sequential(nn.Linear(1, n_n), copy.deepcopy(func))]
        for i in range(1, h_n):
            layer_list.append(nn.Sequential(nn.Linear(n_n, n_n), copy.deepcopy(func)))
        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Linear(n_n, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 0: (N,1) → (N, n_n)
        h = self.layers[0](x)
        outputs = [h]

        # Layers 1 … h_n-1
        for idx in range(1, self.h_n):
            layer_input = outputs[idx - 1]
            for src_idx in self._skip_sources[idx]:
                layer_input = layer_input + outputs[src_idx]
            h = self.layers[idx](layer_input)
            outputs.append(h)
            
        return self.head(outputs[-1])


# ── Convolutional Residual Network ─────────────────────────────────────────────────────────
class ConvResNet(nn.Module):
    """
    Convolutional Residual Network for 1-D regression.

    CNN + Skip(residual)-connections

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
        kernel_size : Conv1d kernel size.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
        func        : activation function.
        connect     : extra-skip window size.  connect=0 → pure ResNet.
    """
    def __init__(self, h_n: int, n_n: int, kernel_size: int, padding: int, stride: int, func: nn.Module, connect: int = 1):
        super().__init__()
        self.h_n = h_n
        self.n_n = n_n
        self.connect = connect
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # ── Connectivity ──────────────────────────────────────────────
        # Shape: (h_n, n_n, n_n).
        # Layer 0: Conv1d(1, n_n, k=1) — single input channel feeds all
        #          n_n filters equally → full.
        # Layers 1+: Conv1d(n_n, n_n, k, p, s) → sparse banded matrix.
        conv_conn = _conv_connections(n_n, kernel_size, padding, stride)
        self._connections: np.ndarray = np.stack([_full_connections(n_n)] + [conv_conn] * max(h_n - 1, 0), axis=0)
        # ── Skip-connection maps ─────────────────────────────────────────
        self._skip_targets: list[list[int]] = []
        for i in range(h_n):
            if i + connect + 2 <= h_n:
                targets = [j for j in range(h_n) if i + 1 < j <= i + connect + 1]
            else:
                targets = []
            self._skip_targets.append(targets)
            
        self._skip_sources: list[list[int]] = [[] for _ in range(h_n)]
        for i, targets in enumerate(self._skip_targets):
            for j in targets:
                self._skip_sources[j].append(i)
        
        # ── Build layers ─────────────────────────────────────────────────
        # Layer 0: Linear(1→n_n) + act (no conv, no residual).
        # Layers 1…h_n-1: conv layer.
        layer_list: list[nn.Module] = []
        layer_list.append(nn.Sequential(nn.Conv1d(1, n_n, kernel_size=1), copy.deepcopy(func)))
        for _ in range(1, h_n):
            layer_list.append(nn.Sequential(nn.Conv1d(n_n, n_n, kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(func)))
        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)       # (B, 1, N)
        # Layer 0: (N,1) → (N, n_n) → (n_n, N) for Conv1d
        h = self.layers[0](x)
        outputs = [h]

        # Layers 1 … h_n-1
        for idx in range(1, self.h_n):
            layer_input = outputs[idx - 1]
            for src_idx in self._skip_sources[idx]:
                layer_input = layer_input + outputs[src_idx]
            h = self.layers[idx](layer_input)
            outputs.append(h)
            
        out = self.head(outputs[-1])
        out = out.permute(0, 2, 1)      # (B, N, 1)
        return out.squeeze(0)