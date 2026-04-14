import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Callable


# ── LayerInfo ─────────────────────────────────────────────────────────
@dataclass
class LayerInfo:
    """
    Describes one layer's visual representation in the graph.

    Attributes:
        n_nodes      : number of nodes (channels / features) in this layer.
        label        : short string shown below the column.
        connectivity : callable(j: int, n_in: int) -> list[int]
                       Given output-node index j and the display-width of the
                       previous layer, return the list of source node indices
                       that j connects from.
        extra_srcs   : list of (src_col_absolute, [src_node_indices]) pairs
                       for skip / residual connections drawn as curved arrows.
    """
    n_nodes     : int
    label       : str
    connectivity: Callable[[int, int], list[int]] = field(default_factory=lambda: (lambda j, n_in: list(range(n_in))))
    extra_srcs  : list[tuple[int, list[int]]] = field(default_factory=list)


# ── Weight extraction helpers ─────────────────────────────────────────
def extract_layer_weights(model: nn.Module) -> list:
    """Extract weights from Linear and Conv1d layers in order."""
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy())
        elif isinstance(module, nn.Conv1d):
            w = module.weight.detach().cpu().numpy()
            weights.append(w.mean(axis=2))
    return weights

# ── Connectivity factories ────────────────────────────────────────────
def _conn_from_matrix(conn_matrix: np.ndarray) -> Callable[[int, int], list[int]]:
    """
    Build a LayerInfo-compatible connectivity callable from one boolean
    (n_n × n_n) slice of model._connections.
 
    connectivity(j, n_in) returns the source indices in [0, n_in) where
    conn_matrix[j, :] is True.
    """
    def _conn(j: int, n_in: int) -> list[int]:
        if j >= conn_matrix.shape[0]:
            return list(range(n_in))
        return np.where(conn_matrix[j, :n_in])[0].tolist()
    return _conn

# ── Subtitle helper ───────────────────────────────────────────────────
def _subtitle_from_layers(infos: list[LayerInfo]) -> str:
    parts: list[str] = []
    seen_conv = seen_fc = seen_skip = False
    for info in infos[1:]:
        if info.extra_srcs:
            seen_skip = True
        if "conv" in info.label:
            seen_conv = True
        elif "fc" in info.label or "out" in info.label:
            seen_fc = True
    if seen_conv:
        parts.append("conv")
    if seen_fc:
        parts.append("fully connected")
    if seen_skip:
        parts.append("skip connections")
    return "  |  ".join(parts) if parts else "custom"


# ── ResNet-aware graph builder ────────────────────────────────────────
def _build_skip_graph_layers(model: nn.Module) -> list[LayerInfo]:
    """
    Build LayerInfo list for models with _skip_sources and _connections.
 
    Column layout (absolute indices):
        col 0          : input stub  (1 node)
        col 1 … h_n    : hidden layers 0 … h_n-1  (n_n nodes each)
        col h_n+1      : output head  (1 node)
 
    For each hidden layer j, connectivity comes from model._connections[j]
    and skip arrows from model._skip_sources[j] (converted to absolute cols).
    """
    h_n = model.h_n
    n_n = model.n_n
 
    # hidden layer k → absolute column k + 1
    skip_sources_abs: list[list[int]] = [[k + 1 for k in model._skip_sources[j]]for j in range(h_n)]
    infos: list[LayerInfo] = []
 
    # col 0 — input stub
    infos.append(LayerInfo(n_nodes=1, label="in"))
    
    # cols 1 … h_n — hidden layers
    for j in range(h_n):
        extra = [(src_col_abs, list(range(n_n)))for src_col_abs in skip_sources_abs[j]]
        infos.append(LayerInfo(n_nodes=n_n, label=f"h{j}\n{n_n}n", connectivity=_conn_from_matrix(model._connections[j]), extra_srcs=extra))
 
    # col h_n+1 — output head
    infos.append(LayerInfo(n_nodes=1, label="out"))
    return infos


# ── Generic model introspection ───────────────────────────────────────
def introspect_model(model: nn.Module) -> tuple[list[LayerInfo], str]:
    """
    Walk a model and build a list of LayerInfo objects for draw_graph.
 
    Priority:
      1. If the model has _skip_sources and layers, use the ResNet-aware path
         (_build_skip_graph_layers), which reads model._connections per layer.
      2. Generic fallback for models without skip connections (FCNN, CNN):
         walk nn.Linear / nn.Conv1d children and read model._connections in
         order to assign the correct connectivity callable to each layer.
 
    Returns:
        layers_info : list[LayerInfo]  (includes the input stub at index 0)
        subtitle    : str
    """
    if hasattr(model, '_skip_sources') and hasattr(model, 'layers'):
        infos = _build_skip_graph_layers(model)
        return infos, _subtitle_from_layers(infos)
 
    # ── Generic introspection (FCNN, CNN) ─────────────────────────────
    layers_info: list[LayerInfo] = [LayerInfo(n_nodes=1, label="in")]
 
    conn_idx   = 0   # index into model._connections along axis 0
    conv_count = 0
    fc_count   = 0
 
    def _process_module(mod: nn.Module) -> None:
        nonlocal conn_idx, conv_count, fc_count
        if isinstance(mod, nn.Conv1d):
            if mod.out_channels == 1:          # output head — not in _connections
                layers_info.append(LayerInfo(n_nodes=1, label="out"))
                return
            conv_count += 1
            k = mod.kernel_size[0]
            p = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
            s = mod.stride[0]  if isinstance(mod.stride,  tuple) else mod.stride
            label = f"conv{conv_count}\n{mod.out_channels}n\nk={k} p={p} s={s}"
            conn = _conn_from_matrix(model._connections[conn_idx])
            conn_idx += 1
            layers_info.append(LayerInfo(n_nodes=mod.out_channels, label=label, connectivity=conn))
        elif isinstance(mod, nn.Linear):
            if mod.out_features == 1:          # output head — not in _connections
                layers_info.append(LayerInfo(n_nodes=1, label="out"))
                return
            fc_count += 1
            label = f"fc{fc_count}\n{mod.out_features}n"
            conn = _conn_from_matrix(model._connections[conn_idx])
            conn_idx += 1
            layers_info.append(LayerInfo(n_nodes=mod.out_features, label=label, connectivity=conn))
        elif isinstance(mod, (nn.Sequential, nn.ModuleList)):
            for child in mod.children():
                _process_module(child)
 
    for name, child in model.named_children():
        _process_module(child)
 
    if layers_info[-1].label != "out" and layers_info[-1].n_nodes != 1:
        layers_info.append(LayerInfo(n_nodes=1, label="out"))
 
    return layers_info, _subtitle_from_layers(layers_info)


# ── draw_graph ────────────────────────────────────────────────────────
def draw_graph(ax: plt.Axes, model: nn.Module, ellipsize_after: int = 8) -> None:
    """
    Draw the network architecture on *ax*.
 
    extra_srcs in LayerInfo stores *absolute* source column indices so arc
    arrows are drawn correctly for any skip distance.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
 
    layers_info, subtitle = introspect_model(model)
 
    weights = extract_layer_weights(model)
    max_weight = max(np.max(np.abs(w)) for w in weights) if weights else 1.0
 
    layer_weight_idx = 0
 
    n_cols = len(layers_info)
    x_pos = np.linspace(0, (n_cols - 1) * 3, n_cols)
 
    def n_shown(ch: int) -> int:
        return min(ch, ellipsize_after)
 
    def y_coords(ch: int) -> np.ndarray:
        n = n_shown(ch)
        return np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
 
    # ── Edges (primary connections) ───────────────────────────────────
    for li in range(1, n_cols):
        info_in = layers_info[li - 1]
        info_out = layers_info[li]
        ys_in = y_coords(info_in.n_nodes)
        ys_out = y_coords(info_out.n_nodes)
        n_in = n_shown(info_in.n_nodes)
        n_out = n_shown(info_out.n_nodes)
 
        weight_matrix = None
        if layer_weight_idx < len(weights):
            weight_matrix = weights[layer_weight_idx]
            layer_weight_idx += 1
 
        for j in range(n_out):
            srcs = info_out.connectivity(j, n_in)
            for src in srcs:
                if 0 <= src < n_in:
                    color = "#000000"
                    alpha = 0.5
                    if weight_matrix is not None:
                        try:
                            w     = weight_matrix[j, src]
                            alpha = (abs(w) / max_weight) * 0.95 + 0.05
                            color = "#d62728" if w < 0 else "#1f77b4"
                        except Exception:
                            pass
                    ax.plot([x_pos[li-1], x_pos[li]], [ys_in[src], ys_out[j]], color=color, lw=1, alpha=alpha, zorder=1)
 
    # ── Edges (skip / residual connections) ───────────────────────────
    for li, info_out in enumerate(layers_info):
        for src_col_abs, skip_srcs in info_out.extra_srcs:
            info_skip = layers_info[src_col_abs]
            ys_skip = y_coords(info_skip.n_nodes)
            ys_out = y_coords(info_out.n_nodes)
            n_skip = n_shown(info_skip.n_nodes)
            n_out = n_shown(info_out.n_nodes)
 
            skip_dist = li - src_col_abs
            rad = 0.2 + 0.1 * skip_dist
 
            for j, src in enumerate(skip_srcs):
                if 0 <= src < n_skip and j < n_out:
                    ax.annotate("", xy=(x_pos[li],ys_out[j]), xytext=(x_pos[src_col_abs],ys_skip[src]),
                                arrowprops=dict(arrowstyle="->", color="#c0c0c0", lw=1, alpha=0.5, connectionstyle=f"arc3,rad={rad:.2f}"), zorder=1)
 
    # ── Nodes ─────────────────────────────────────────────────────────
    node_r = 0.22
    dot_r  = 0.06
    for li, info in enumerate(layers_info):
        ys = y_coords(info.n_nodes)
        is_ellipsized = info.n_nodes > ellipsize_after
        for i, y in enumerate(ys):
            is_dots = is_ellipsized and i == len(ys) - 1
            if is_dots:
                spacing = node_r * 0.9
                for dy in (-spacing, 0, spacing):
                    ax.add_patch(plt.Circle((x_pos[li], y + dy), dot_r, color="#444444", zorder=2, linewidth=0))
            else:
                ax.add_patch(plt.Circle((x_pos[li], y), node_r,facecolor="#5B9BD5", zorder=2,linewidth=0.8, edgecolor="white"))
 
    # ── Column labels ─────────────────────────────────────────────────
    max_shown = max(n_shown(info.n_nodes) for info in layers_info)
    y_bot     = -(max_shown / 2) - 0.9
 
    for li, info in enumerate(layers_info):
        ax.text(x_pos[li], y_bot, info.label, ha="center", va="top", fontsize=7, color="#444444")
 
    ax.set_title(f"Network graph  ({subtitle})", fontsize=8, pad=4)
    margin = 0.2
    ax.set_xlim(x_pos[0] - margin, x_pos[-1] + margin)
    ax.set_ylim(y_bot - margin, max_shown / 2 + margin)
 
 
# ── Animation factory ─────────────────────────────────────────────────
def make_animation(
    model: nn.Module,
    snapshots: list,
    mse_losses: list,
    bce_losses: list,
    x_np: np.ndarray,
    y_np: np.ndarray,
    epochs: int,
    title: str,
    pred_color: str,
    mse_color: str,
    bce_color: str,
    file_type: str="",
    ifsave: bool=False
) -> tuple:
    """
    Build a 2×2 animated figure:
        [top-left]     curve fit (animated)
        [top-right]    network graph (static)
        [bottom-left]  MSE loss curve (animated)
        [bottom-right] binary cross-entropy loss curve (animated)
 
    Returns (fig, ani).
    """
    epoch_vals = np.arange(1, epochs + 1)
 
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
 
    ax_fit   = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_mse   = fig.add_subplot(gs[1, 0])
    ax_bce   = fig.add_subplot(gs[1, 1])
 
    # ── Top-left: curve fit ───────────────────────────────────────────
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa",
                   label="Data", zorder=1)
    ax_fit.set_xlim(np.min(x_np) * 1.05, np.max(x_np) * 1.05)
    ax_fit.set_ylim(np.min(y_np) * 1.2,  np.max(y_np) * 1.2)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.set_title("Curve fit")
 
    # ── Top-right: network graph (static) ────────────────────────────
    draw_graph(ax_graph, model)
 
    # ── Bottom-left: MSE loss ─────────────────────────────────────────
    ax_mse.set_xlim(1, epochs)
    ax_mse.set_ylim(max(min(mse_losses) * 0.5, 1e-9), max(mse_losses) * 1.1)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE Loss")
    ax_mse.set_title("MSE Loss")
 
    # ── Bottom-right: BCE loss ────────────────────────────────────────
    ax_bce.set_xlim(1, epochs)
    ax_bce.set_ylim(max(min(bce_losses) * 0.5, 1e-9), max(bce_losses) * 1.1)
    ax_bce.set_yscale("log")
    ax_bce.set_xlabel("Epoch")
    ax_bce.set_ylabel("Binary Cross Entropy Loss")
    ax_bce.set_title("Binary Cross Entropy Loss")
 
    # ── Dynamic artists ───────────────────────────────────────────────
    (line_pred,) = ax_fit.plot([], [], lw=2, color=pred_color, zorder=2, label="Prediction")
    fit_title = ax_fit.set_title("")
    ax_fit.legend(loc="upper right", fontsize=8)
 
    (line_mse,) = ax_mse.plot([], [], lw=1.5, color=mse_color)
    dot_mse, = ax_mse.plot([], [], "o", color=pred_color, ms=6, zorder=3)
 
    (line_bce,) = ax_bce.plot([], [], lw=1.5, color=bce_color)
    dot_bce, = ax_bce.plot([], [], "o",    color=pred_color, ms=6, zorder=3)
 
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.35)
 
    def update(frame_idx: int):
        epoch, mse_val, bce_val, y_pred = snapshots[frame_idx]
        mask = epoch_vals <= epoch
 
        line_pred.set_data(x_np, y_pred)
        fit_title.set_text(f"Epoch {epoch:>4d}")
 
        line_mse.set_data(epoch_vals[mask], np.array(mse_losses)[mask])
        dot_mse.set_data([epoch], [mse_val])
 
        line_bce.set_data(epoch_vals[mask], np.array(bce_losses)[mask])
        dot_bce.set_data([epoch], [bce_val])
 
        return line_pred, fit_title, line_mse, dot_mse, line_bce, dot_bce
    
    ani = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=80,
        blit=True,
        repeat=False
        )
    if ifsave:
        name = title.split("—", maxsplit=1)[0]
        name = "".join(name.split())
        print(f"{name}.{file_type} … ", end="")
        if file_type == 'gif':
            ani.save(f"{name}.{file_type}", writer="pillow", fps=15)
        else:    
            update(len(snapshots)-1)
            fig.savefig(f"{name}.{file_type}")
        print("Saved")
    return fig, ani