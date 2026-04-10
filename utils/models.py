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
            conv_layers.extend([
                nn.Conv1d(in_ch, n_n, kernel_size=kernel_size, padding=padding, stride=stride),
                copy.deepcopy(func),
            ])
            in_ch = n_n
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, 1)
        # Conv1d expects (batch, channels, length) -> reshape to (1, 1, N)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, 1)
        x = x.permute(0, 2, 1)  # (B, 1, N)
        out = self.conv(x)
        out = self.head(out)
        out = out.permute(0, 2, 1)  # (B, N, 1)
        return out.squeeze(0)


# ── Dense Highway Neural Network ──────────────────────────────────────────────────────────
class _DenseHighwayLayer(nn.Module):
    """
    One layer of a Dense-HNN.

    Receives the concatenation of ALL previous hidden states (including the
    input projection), so every node in every earlier layer feeds directly
    into both the transform branch H and the gate T.  The carry term reuses
    only the immediately preceding layer's output (last n_n slice of the
    concatenated input) so that dimensions stay consistent.

    Args:
        in_size   : total width of the concatenated input  (= layer_index × n_n).
        n_n       : output width (fixed across all layers).
        func      : activation for the H branch.
        bias_init : initial bias for T; negative → biased toward carry at init.
    """
    def __init__(self, in_size: int, n_n: int, func: nn.Module, bias_init: float = -2.0):
        super().__init__()
        self.n_n  = n_n
        self.H    = nn.Linear(in_size, n_n)
        self.T    = nn.Linear(in_size, n_n)
        self.func = copy.deepcopy(func)
        nn.init.constant_(self.T.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cat_x : (N, in_size)  — concatenation of all previous outputs
        H    = self.func(self.H(x))
        T    = torch.sigmoid(self.T(x))
        prev = x[..., -self.n_n:]          # output of the previous layer
        return H * T + prev * (1.0 - T)        # (N, n_n)


class DenseHNN(nn.Module):
    """
    Dense Highway Neural Network for 1-D regression.

    Every hidden node receives connections from every node in every previous
    hidden layer (dense connectivity), while the highway gate controls how
    much of the new transform vs. the previous layer's output is kept.

    Architecture:
        Linear(1 → n_n) + func                          — input projection
        h_n × _DenseHighwayLayer                        — dense highway stack
            layer i input  = concat(h_0, h_1, …, h_{i-1})   width = i × n_n
            layer i output = H·T + h_{i-1}·(1-T)             width = n_n
        Linear(n_n → 1)                                 — output head

    The input and output layers connect only to their immediate neighbour
    (input proj → layer 0, last highway layer → head), matching the spec.

    Args:
        h_n       : number of dense highway layers.
        n_n       : number of nodes per layer (fixed width throughout).
        func      : activation for H branches and the input projection.
        bias_init : transform-gate bias init (more negative → more carry).
    """
    def __init__(
        self,
        h_n:       int,
        n_n:       int,
        func:      nn.Module,
        bias_init: float = -2.0,
    ):
        super().__init__()
        self.h_n       = h_n
        self.n_n       = n_n
        self.bias_init = bias_init

        self.input_proj = nn.Sequential(
            nn.Linear(1, n_n),
            copy.deepcopy(func),
        )
        # layer i sees i+1 previous outputs (input_proj + i highway outputs)
        self.highway = nn.ModuleList([
            _DenseHighwayLayer((i + 1) * n_n, n_n, func, bias_init)
            for i in range(h_n)
        ])
        self.head = nn.Linear(n_n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h       = self.input_proj(x)    # (N, n_n)
        outputs = [h]
        for layer in self.highway:
            cat = torch.cat(outputs, dim=-1)    # (N, i*n_n)
            h   = layer(cat)                    # (N, n_n)
            outputs.append(h)
        return self.head(outputs[-1])           # (N, 1)

    def graph_layers(self) -> list:
        """
        Return a list of LayerInfo descriptors for draw_graph.

        Layout:
            col 0          : input stub  (1 node)
            col 1          : input_proj  (n_n nodes, full fan-out from col 0)
            cols 2 … h_n+1 : highway layers (n_n nodes each)
                             primary edge  : full fan-in from the previous col
                             extra_srcs    : full fan-in from every earlier col
                             (this represents the dense concatenation)
            col h_n+2      : output head  (1 node, full fan-in from last highway col)
        """
        from utils.animation import LayerInfo, full_connectivity

        infos: list[LayerInfo] = []

        # col 0 — input stub
        infos.append(LayerInfo(n_nodes=1, label="in",
                               connectivity=full_connectivity))

        # col 1 — input projection
        infos.append(LayerInfo(n_nodes=self.n_n, label=f"proj\n{self.n_n}n",
                               connectivity=full_connectivity))

        # cols 2 … h_n+1 — highway layers
        for i in range(self.h_n):
            col_idx = i + 2   # absolute column index of this highway layer
            # extra_srcs: connect back to every column before the previous one
            # layer_offset is relative to col_idx, so offset = src_col - col_idx
            extra = [
                (src_col - col_idx, list(range(self.n_n)))
                for src_col in range(1, col_idx - 1)   # all cols except immediate prev
            ]
            infos.append(LayerInfo(
                n_nodes=self.n_n,
                label=f"hw{i+1}\n{self.n_n}n",
                connectivity=full_connectivity,   # primary: from previous col
                extra_srcs=extra,
            ))

        # last col — output head
        infos.append(LayerInfo(n_nodes=1, label="out",
                               connectivity=full_connectivity))
        return infos


# ── Convolutional Highway Neural Network ───────────────────────────────────────────────────
class _ConvHighwayLayer(nn.Module):
    """
    One layer of a Conv-HNN.

    The hidden state at each step is an (n_n × N_seq) feature map.
    All previous hidden states are stacked into a 3-D tensor of shape
    (i_layers, n_n, N_seq) and treated as a 2-D spatial map where:
        axis 0  (height) = layer axis   — which hidden layer
        axis 1  (width)  = node axis    — which node within a layer

    A Conv2d with kernel_size=(kernel_size, kernel_size) is applied over
    this (layer × node) plane independently for each sequence position,
    mixing information across both earlier layers and neighbouring nodes
    with the same sparsity pattern as the CNN's Conv1d.

    AdaptiveAvgPool2d collapses the variable layer-axis height back to 1
    so the output is always (N_seq, n_n) regardless of depth.

    Args:
        n_n         : nodes per layer (= Conv2d spatial width, fixed).
        kernel_size : kernel size on both the layer and node axes.
        padding     : padding on both axes (keeps node axis width = n_n).
        stride      : stride on both axes.
        func        : activation for the H branch.
        bias_init   : initial bias for T gate.
    """
    def __init__(
        self,
        n_n:         int,
        kernel_size: int,
        padding:     int,
        stride:      int,
        func:        nn.Module,
        bias_init:   float = -2.0,
    ):
        super().__init__()
        self.n_n  = n_n
        self.func = copy.deepcopy(func)
        self.conv_H = nn.Conv1d(n_n, n_n, kernel_size, padding=padding, stride=stride)
        self.conv_T = nn.Conv1d(n_n, n_n, kernel_size, padding=padding, stride=stride)
        self.pool   = nn.AdaptiveAvgPool2d((1, n_n))
        nn.init.constant_(self.conv_T.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H = self.func(self.conv_H(x))
        T = torch.sigmoid(self.conv_T(x))
        return H * T + x * (1.0 - T)


class ConvHNN(nn.Module):
    """
    Convolutional Highway Neural Network for 1-D regression.

    Combines CNN-style 2-D sparse connectivity with highway carry gates.
    At each layer the full stack of previous hidden states forms a
    (layer × node) 2-D map; a Conv2d kernel mixes across both dimensions
    with the same kernel_size / padding / stride as the plain CNN, so the
    connectivity pattern is identical but gated.

    Architecture:
        Linear(1 → n_n) + func                          — input projection
        h_n × _ConvHighwayLayer                         — conv highway stack
            layer i input  = stack(h_0, …, h_{i-1})    shape (i, n_n, N_seq)
            layer i output = H·T + h_{i-1}·(1-T)       shape (N_seq, n_n)
        Linear(n_n → 1)                                 — output head

    Args:
        h_n         : number of convolutional highway layers.
        n_n         : nodes / filters per layer.
        kernel_size : Conv2d kernel size (applied on both layer and node axes).
        padding     : Conv2d padding (both axes).
        stride      : Conv2d stride (both axes).
        func        : activation for H branches and the input projection.
        bias_init   : transform-gate bias init (more negative → more carry).
    """
    def __init__(
        self,
        h_n:         int,
        n_n:         int,
        kernel_size: int,
        padding:     int,
        stride:      int,
        func:        nn.Module,
        bias_init:   float = -2.0,
    ):
        super().__init__()
        self.h_n         = h_n
        self.n_n         = n_n
        self.kernel_size = kernel_size
        self.padding     = padding
        self.stride      = stride
        self.bias_init   = bias_init

        self.input_proj = nn.Sequential(
            nn.Linear(1, n_n),
            copy.deepcopy(func),
        )
        self.highway = nn.ModuleList([
            _ConvHighwayLayer(n_n, kernel_size, padding, stride, func, bias_init)
            for _ in range(h_n)
        ])
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x).T.unsqueeze(0)   # (1, n_n, N)
        for layer in self.highway:
            h = layer(h)
        return self.head(h).squeeze(0).T        # (N, 1)

    def graph_layers(self) -> list:
        """
        Return a list of LayerInfo descriptors for draw_graph.

        Layout:
            col 0          : input stub  (1 node)
            col 1          : input_proj  (n_n nodes, full fan-out from col 0)
            cols 2 … h_n+1 : highway layers (n_n nodes each)
                             primary edge  : conv1d-sparse from the previous col
                             extra_srcs    : conv1d-sparse from every earlier col
                             (mirrors the Conv2d kernel that spans both axes)
            col h_n+2      : output head  (1 node, full fan-in from last highway col)
        """
        from utils.animation import LayerInfo, full_connectivity, conv1d_connectivity

        sparse = conv1d_connectivity(self.kernel_size, self.padding, self.stride)
        infos: list[LayerInfo] = []

        # col 0 — input stub
        infos.append(LayerInfo(n_nodes=1, label="in",
                               connectivity=full_connectivity))

        # col 1 — input projection (full fan-out from single input)
        infos.append(LayerInfo(n_nodes=self.n_n, label=f"proj\n{self.n_n}n",
                               connectivity=full_connectivity))

        # cols 2 … h_n+1 — highway layers
        for i in range(self.h_n):
            col_idx = i + 2
            # extra_srcs: conv-sparse connection back to every column before the previous
            extra = [
                (src_col - col_idx, list(range(self.n_n)))
                for src_col in range(1, col_idx - 1)
            ]
            infos.append(LayerInfo(
                n_nodes=self.n_n,
                label=f"hw{i+1}\n{self.n_n}n",
                connectivity=sparse,      # primary: conv-sparse from previous col
                extra_srcs=extra,
            ))

        # last col — output head
        infos.append(LayerInfo(n_nodes=1, label="out",
                               connectivity=full_connectivity))
        return infos