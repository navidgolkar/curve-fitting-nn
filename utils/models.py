import torch
import torch.nn as nn
import numpy as np
import copy

from utils.parameters import ModelParams, apply_seed, check_parameters

# Connectivity helpers --------------------------------------------------------
def _build_dense_incoming(layer_sizes: list[int]) -> list[list[list[tuple[int, int]]]]:
    """Dense _incoming: every node in L receives from every node in L-1."""
    n = len(layer_sizes)
    inc: list[list[list[tuple[int, int]]]] = [
        [[] for _ in range(layer_sizes[L])] for L in range(n)
    ]
    for L in range(1, n):
        for tgt_n in range(layer_sizes[L]):
            for src_n in range(layer_sizes[L - 1]):
                inc[L][tgt_n].append((L - 1, src_n))
    return inc

def _full_connections(n_out: int, n_in: int) -> np.ndarray:
    """
    All-True (n_out × n_in) matrix for a Linear(n_in → n_out) layer.
    Entry [out, in] = True  ↔  out-node receives from in-node.
    """
    return np.ones((n_out, n_in), dtype=bool)
 
 
def _conv_connections(n_out: int, n_in: int, kernel_size: int, padding: int, stride: int) -> np.ndarray:
    """
    Sparse (n_out × n_in) connectivity matrix for Conv1d(n_in, n_out, ...).
    Entry [out_filter, in_filter] = True  ↔  out_filter's receptive field
    overlaps with in_filter.
    """
    conn = np.zeros((n_out, n_in), dtype=bool)
    for j in range(n_out):
        i_min = j * stride - padding
        i_max = i_min + kernel_size - 1
        for i in range(n_in):
            if i_min <= i <= i_max:
                conn[j, i] = True
    return conn

# Fully Connected Neural Network ----------------------------------------------
class FCNN(nn.Module):
    """
    Fully-connected neural network built from ``params.layer_sizes``.
    
    Each adjacent layer pair ``(layer_sizes[i], layer_sizes[i+1])`` becomes one
    ``nn.Linear`` followed by ``activation_functions[i]``, except the final
    transition which has no activation (the last entry in
    ``activation_functions`` is applied to the last hidden→output edge, so
    pass ``nn.Identity()`` there if you want a linear output).
    
    Args:
        params : ModelParams
            * ``layer_sizes``         — width of every layer; can vary freely.
            * ``activation_functions``— ``number of layers - 1`` activations, one per
              transition.
            * ``seed``, ``device``, and all training fields.
    """
    def __init__(self, params: ModelParams):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        sizes = params.layer_sizes
        funcs = params.activation_functions
        n_layers = len(sizes)
        
        # Connectivity: one (out × in) matrix per transition
        self._connections: list[np.ndarray] = [_full_connections(sizes[i + 1], sizes[i]) for i in range(n_layers - 1)]
        
        layers: list[nn.Module] = []
        for i in range(n_layers - 2):
            layers.extend([nn.Linear(sizes[i], sizes[i + 1]), copy.deepcopy(funcs[i])])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(sizes[-2], sizes[-1])
        
        params._is_resnet = False
        params._skip_connections = [[] for _ in range(n_layers)]
        params._incoming = _build_dense_incoming(sizes)
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return self.head(out)


# Convolutional Neural Network ------------------------------------------------
class CNN(nn.Module):
    """
    1-D convolutional neural network built from ``params.layer_sizes``.
    
    ``layer_sizes[0]`` is treated as 1 input channel; the first Conv1d always
    uses ``kernel_size=1`` to fan out to ``layer_sizes[1]`` filters.
    Subsequent hidden transitions use the supplied *kernel_size* / *padding* /
    *stride*.  The final layer is a ``Conv1d(..., 1, kernel_size=1)`` head
    that collapses back to a single output channel.
    
    Args:
        params      : ModelParams — ``layer_sizes``, ``activation_functions``,
                      ``seed``, ``device``, and all training fields.
        kernel_size : Conv1d kernel size for hidden→hidden transitions.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
    """
    def __init__(self, params: ModelParams, kernel_size: int, padding: int, stride: int):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        sizes = params.layer_sizes
        funcs = params.activation_functions
        n_layers = len(sizes)
        
        # Connectivity matrices (one per hidden transition)
        self._connections: list[np.ndarray] = []
        self._connections.append(_conv_connections(sizes[1], sizes[0], kernel_size=1, padding=0, stride=1))
        for i in range(1, n_layers-2):
            self._connections.append(_conv_connections(sizes[i + 1], sizes[i], kernel_size, padding, stride))
        self._connections.append(_conv_connections(sizes[-1], sizes[-2], kernel_size=1, padding=0, stride=1))
        
        # Build conv body
        conv_layers: list[nn.Module] = [nn.Conv1d(sizes[0], sizes[1], kernel_size=1), copy.deepcopy(funcs[0])]
        for i in range(1, n_layers-2):
            conv_layers.extend([nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(funcs[i])])
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(sizes[-2], sizes[-1], kernel_size=1)
        
        params._is_resnet = False
        params._skip_connections = [[] for _ in range(n_layers)]
        params._incoming = _build_dense_incoming(sizes)
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instead of permute you can Transpose (x.T) but to be consistent with ConvResNet which needs permute I used permute here as well
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x   = x.permute(0, 2, 1)       # (B, 1, N)
        out = self.conv(x)
        out = self.head(out)
        out = out.permute(0, 2, 1)      # (B, N, 1)
        return out.squeeze(0)

# Dense Residual Network ------------------------------------------------------
class DenseResNet(nn.Module):
    """
    Dense Residual Network built from ``params.layer_sizes``. Residual summation
    is done after activation function.
    
    Every hidden layer sends a long-range skip to every later hidden layer
    (layer i → layer j for all j > i + 1).  Because skips are parameter-free
    additions, all *hidden* layers must have the same width (``layer_sizes[1]``
    through ``layer_sizes[-2]`` must be equal).  Input and output widths can
    differ freely.
    
    Skip-connection rule (0-indexed over hidden layers):
        Layer i skips to every j where j > i + 1.
    
    Args:
        params : ModelParams — ``layer_sizes``, ``activation_functions``,
                 ``seed``, ``device``, and all training fields.
    
    Raises:
        ValueError: If hidden layer widths are not all equal.
    """
    def __init__(self, params: ModelParams):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        sizes = params.layer_sizes
        funcs = params.activation_functions
        self.n_layers = len(sizes)
        
        if len(set(sizes[1:-1])) > 1:
            raise ValueError(f"DenseResNet requires all hidden layers to have the same width (needed for parameter-free residual addition), got hidden sizes {sizes[1:-1]}.")
        
        # Connectivity: one matrix per hidden transition
        self._connections: list[np.ndarray] = [_full_connections(sizes[i + 1], sizes[i]) for i in range(len(sizes) - 1)]
        
        # Skip-connection maps
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for j in range(1, self.n_layers - 1):       # j: hidden layers in full-layer index
            for i in range(1, j - 1):           # i: all earlier hidden layers (skip gap > 1)
                self._skip_sources[j].append(i)
        
        # Layer 0: input → first hidden
        layer_list: list[nn.Module] = [nn.Sequential(nn.Linear(sizes[0], sizes[1]), copy.deepcopy(funcs[0]))]
        
        for i in range(1, self.n_layers - 2):
            layer_list.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), copy.deepcopy(funcs[i])))
        self.layers = nn.ModuleList(layer_list)
        self.head   = nn.Linear(sizes[-2], sizes[-1])
        
        params._is_resnet = True
        params._skip_connections = self._skip_sources
        params._incoming = _build_dense_incoming(sizes)
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        layer_input = x
        for idx in range(self.n_layers - 2):
            outputs.append(self.layers[idx](layer_input))
            layer_input = outputs[-1]
            for src_idx in self._skip_sources[idx+1]:
                layer_input = layer_input + outputs[src_idx]
        return self.head(layer_input)

# Convolutional Residual Network ----------------------------------------------
class ConvResNet(nn.Module):
    """
    Convolutional Residual Network built from ``params.layer_sizes``. Residual
    summation is done after activation function.
    
    Skip-connection rule (0-indexed over hidden layers):
        Layer i sends a skip to every j where i+1 < j ≤ i+connect+1,
        BUT ONLY when i + connect + 2 ≤ number of layers - 2.
    
    Because skips are parameter-free channel additions, all *hidden* layers
    must share the same filter count (``layer_sizes[1:-1]`` must all be equal).
    
    Args:
        params      : ModelParams — ``layer_sizes``, ``activation_functions``,
                      ``seed``, ``device``, and all training fields.
        kernel_size : Conv1d kernel size for hidden→hidden transitions.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
        connect     : skip window size.  ``connect=0`` → no skips (pure CNN).
    
    Raises:
        ValueError: If hidden layer widths are not all equal.
    """
    def __init__(self, params: ModelParams, kernel_size: int, padding: int, stride: int, connect: int = 1):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.connect = connect
        
        sizes = params.layer_sizes
        funcs = params.activation_functions
        self.n_layers = len(sizes)
        
        if len(set(sizes[1:-1])) > 1:
            raise ValueError(f"ConvResNet requires all hidden layers to have the same filter count (needed for parameter-free residual addition), got hidden sizes {sizes[1:-1]}.")
        
        # Connectivity matrices
        self._connections: list[np.ndarray]
        self._connections.append(_conv_connections(sizes[1], sizes[0], kernel_size=1, padding=0, stride=1))
        for i in range(1, self.n_layers - 2):
            self._connections.append(_conv_connections(sizes[i + 1], sizes[i], kernel_size, padding, stride))
        self._connections.append(_conv_connections(sizes[-1], sizes[-2], kernel_size=1, padding=0, stride=1))
        
        # Skip-connection maps
        self._skip_targets: list[list[int]] = [[] for _ in range(self.n_layers)]
        for i in range(self.n_layers - 2):                   # i: hidden-local source index
            if i + connect + 2 <= self.n_layers - 2:
                for j in range(self.n_layers - 2):
                    if i + 1 < j <= i + connect + 1:
                        self._skip_targets[i+1].append(j+1)
            
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for i, targets in enumerate(self._skip_targets):
            for j in targets:
                self._skip_sources[j].append(i)
                
        # Build layers
        layer_list: list[nn.Module] = [nn.Sequential(nn.Conv1d(1, sizes[1], kernel_size=1), copy.deepcopy(funcs[0]))]
        for i in range(1, self.n_layers - 2):
            layer_list.append(nn.Sequential(nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(funcs[i])))
        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Conv1d(sizes[self.n_layers - 2], 1, kernel_size=1)
        
        params._is_resnet = True
        params._skip_connections = self._skip_sources
        params._incoming = _build_dense_incoming(sizes)
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)       # (B, 1, N)
        outputs = []
        layer_input = x
        for idx in range(self.n_layers - 2):
            outputs.append(self.layers[idx](layer_input))
            layer_input = outputs[-1]
            for src_idx in self._skip_sources[idx+1]:
                layer_input = layer_input + outputs[src_idx]
        out = self.head(layer_input)
        out = out.permute(0, 2, 1)      # (B, N, 1)
        return out.squeeze(0)
    
# Custom Graph Network --------------------------------------------------------
class CustomNet(nn.Module):
    """
    Arbitrary feedforward network defined by an explicit node-connection graph.
    
    **Adjacent edges** (``tgt_layer == src_layer + 1``) pass through a masked
    ``nn.Linear``.  Inactive connections (not listed in *nodes*) are zeroed
    and stay zeroed throughout training.
    
    **Skip / residual edges** (``tgt_layer > src_layer + 1``) are parameter-free
    tensor additions — the source layer's post-activation output is added to the
    target layer's input before its linear and activation.  Requires
    ``layer_sizes[src] == layer_sizes[tgt]``.
    
    Args:
        params : ModelParams
            * ``layer_sizes``, ``activation_functions``.
            * ``_pruned`` — pre-declared pruned edges, applied at construction.
            * ``seed``, ``device``, and all training fields.
        nodes : list[list[list[tuple[int, int]]]]
            ``nodes[src_layer][src_node]`` = ``[(tgt_layer, tgt_node), ...]``
            defining every active edge.
    """
    def __init__(self, params: ModelParams, nodes:  list[list[list[tuple[int, int]]]]) -> None:
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.layer_sizes = list(params.layer_sizes)
        self.n_layers = len(self.layer_sizes)
        funcs = params.activation_functions
        
        # Parse nodes ---------------------------------------------------------
        adjacent_active: dict[int, set[tuple[int, int]]] = {}
        skip_pairs: set[tuple[int, int]] = set()
        
        incoming: list[list[list[tuple[int, int]]]] = [[[] for _ in range(self.layer_sizes[L])] for L in range(self.n_layers)]
        
        for src_layer, src_nodes_list in enumerate(nodes):
            for src_node, targets in enumerate(src_nodes_list):
                for tgt_layer, tgt_node in targets:
                    if tgt_layer <= src_layer:
                        raise ValueError(f"Edge ({src_layer},{src_node})→({tgt_layer},{tgt_node}): tgt_layer must be > src_layer.")
                    if tgt_layer >= self.n_layers:
                        raise ValueError(f"Edge targets layer {tgt_layer} but only {self.n_layers} layers are defined.")
                    if src_node >= self.layer_sizes[src_layer]:
                        raise ValueError(f"src_node {src_node} out of range for layer {src_layer} (size {self.layer_sizes[src_layer]}).")
                    if tgt_node >= self.layer_sizes[tgt_layer]:
                        raise ValueError(f"tgt_node {tgt_node} out of range for layer {tgt_layer} (size {self.layer_sizes[tgt_layer]}).")
        
                    incoming[tgt_layer][tgt_node].append((src_layer, src_node))
                    
                    if tgt_layer == src_layer + 1:
                        adjacent_active.setdefault(src_layer, set()).add((tgt_node, src_node))
                    else:
                        if self.layer_sizes[src_layer] != self.layer_sizes[tgt_layer]:
                            raise ValueError(f"Skip edge ({src_layer})→({tgt_layer}): layer_sizes must match ({self.layer_sizes[src_layer]} ≠ {self.layer_sizes[tgt_layer]}).")
                        skip_pairs.add((src_layer, tgt_layer))
        
        self._incoming = incoming
        
        # Adjacent masks & linears --------------------------------------------
        active_masks: list[torch.Tensor] = []
        for i in range(self.n_layers - 1):
            mask = torch.zeros(self.layer_sizes[i + 1], self.layer_sizes[i], dtype=torch.bool)
            for (tgt_n, src_n) in adjacent_active.get(i, set()):
                mask[tgt_n, src_n] = True
            active_masks.append(mask)
        
        self._active_mask: list[torch.Tensor] = []
        for i, mask in enumerate(active_masks):
            self.register_buffer(f"_mask_{i}", mask)
            self._active_mask.append(getattr(self, f"_mask_{i}"))
            
        self._linears = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=True) for i in range(self.n_layers - 1)])
        
        # Skip maps -----------------------------------------------------------
        self._skip_targets: list[list[int]] = [[] for _ in range(self.n_layers)]
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for (src_l, tgt_l) in skip_pairs:
            self._skip_targets[src_l].append(tgt_l)
            self._skip_sources[tgt_l].append(src_l)
        for lst in self._skip_targets:
            lst.sort()
        for lst in self._skip_sources:
            lst.sort()
        self._funcs = nn.ModuleList()
        self._funcs.extend([copy.deepcopy(f) for f in funcs])
        self._funcs.append(nn.Identity())
        
        # Pruning -------------------------------------------------------------
        self._pruned: set[tuple[int, int, int, int]] = set()
        self._apply_masks()
        for edge in params._pruned:
            self.add_pruned(*edge)
        
        # Write back to params ------------------------------------------------
        params._is_resnet        = bool(skip_pairs)
        params._skip_connections = self._skip_sources
        params._incoming         = self._incoming
        
        self.to(params.device)
    
    def _apply_masks(self) -> None:
        with torch.no_grad():
            for lin, mask in zip(self._linears, self._active_mask):
                lin.weight.data[~mask] = 0.0
                dead = ~mask.any(dim=1)
                if dead.any():
                    lin.bias.data[dead] = 0.0
    def add_pruned(self, src_layer: int, src_node: int, tgt_layer: int, tgt_node: int) -> None:
        """Mark an edge as pruned.  Adjacent edges are weight-zeroed;
        skip edges are removed from the routing maps."""
        if tgt_layer <= src_layer:
            raise ValueError(f"tgt_layer must be > {src_layer}.")
        if src_layer >= self.n_layers - 1:
            raise ValueError(f"src_layer {src_layer} out of range.")
        if src_node >= self.layer_sizes[src_layer]:
            raise ValueError(f"src_node {src_node} out of range.")
        if tgt_node >= self.layer_sizes[tgt_layer]:
            raise ValueError(f"tgt_node {tgt_node} out of range.")
        
        edge = (src_layer, src_node, tgt_layer, tgt_node)
        self._pruned.add(edge)
        self.params._pruned.add(edge)
        
        if tgt_layer == src_layer + 1:
            mask = self._active_mask[src_layer]
            mask[tgt_node, src_node] = False
            with torch.no_grad():
                lin = self._linears[src_layer]
                lin.weight.data[tgt_node, src_node] = 0.0
                if not mask[tgt_node].any():
                    lin.bias.data[tgt_node] = 0.0
        
        else:
            if tgt_layer in self._skip_targets[src_layer]:
                self._skip_targets[src_layer].remove(tgt_layer)
            if src_layer in self._skip_sources[tgt_layer]:
                self._skip_sources[tgt_layer].remove(src_layer)
            self.params._skip_connections = self._skip_sources
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._apply_masks()
        outputs = []
        layer_input = x
        for L in range(self.n_layers-1):
            outputs.append(self._funcs[L](self._linears[L](layer_input)))
            layer_input = outputs[-1]
            for src_l in self._skip_sources[L+1]:
                layer_input = layer_input + outputs[src_l]
        return outputs[-1]