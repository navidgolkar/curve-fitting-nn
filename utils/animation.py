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
        n_nodes        : number of nodes (channels / features) in this layer.
        label          : short string shown below the column, e.g. "conv1\n15ch".
        connectivity   : callable(j: int, n_in: int) -> list[int]
                         Given output-node index j and the display-width of the
                         previous layer, return the list of source node indices
                         that j connects from.  Return range(n_in) for full
                         fan-in (Linear, input stub), or a sparse list for
                         Conv1d with arbitrary kernel / stride / padding.
        extra_srcs     : additional (layer_offset, [src_indices]) pairs used by
                         skip-connection architectures (Highway, DenseNet).
                         layer_offset is negative: -1 = layer before previous, etc.
    """
    n_nodes    : int
    label      : str
    connectivity: Callable[[int, int], list[int]] = field(
        default_factory=lambda: (lambda j, n_in: list(range(n_in)))
    )
    extra_srcs : list[tuple[int, list[int]]] = field(default_factory=list)
    
def extract_layer_weights(model):
    """Extract weights from Linear and Conv1d layers."""
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy())
        elif isinstance(module, nn.Conv1d):
            w = module.weight.detach().cpu().numpy()
            weights.append(w.mean(axis=2))  # average kernel dimension
    return weights


# ── Connectivity factories ─────────────────────────────────────────────
def full_connectivity(j: int, n_in: int) -> list[int]:
    """Every output node connects to every input node (Linear layers)."""
    return list(range(n_in))

def conv1d_connectivity(kernel_size: int, padding: int, stride: int
                        ) -> Callable[[int, int], list[int]]:
    """
    Returns a connectivity function that reflects the actual receptive field
    of a Conv1d layer with the given kernel_size, padding, and stride.

    For output node j the input positions are:
        src = j * stride - padding + k   for k in range(kernel_size)
    Only positions in [0, n_in) are kept (boundary clamping = no padding nodes).
    """
    def _conn(j: int, n_in: int) -> list[int]:
        srcs = []
        for k in range(kernel_size):
            src = j * stride - padding + k
            if 0 <= src < n_in:
                srcs.append(src)
        return srcs
    return _conn

# ── Model introspection ───────────────────────────────────────────────
def _subtitle_from_layers(infos: list[LayerInfo]) -> str:
    """Build a compact subtitle string from layer types present."""
    parts: list[str] = []
    seen_conv  = False
    seen_fc    = False
    seen_skip  = False
    for info in infos[1:]:          # skip input stub
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


def introspect_model(model: nn.Module) -> tuple[list[LayerInfo], str]:
    """
    Walk a model's direct children (and their children for named sub-modules)
    and build a list of LayerInfo objects for draw_graph.

    If the model implements graph_layers() -> list[LayerInfo], that is used
    directly and generic introspection is skipped entirely.  This is the
    preferred path for custom architectures (DenseHNN, ConvHNN, etc.) that
    cannot be faithfully described by walking named_children alone.

    Generic fallback supports:
        nn.Linear      -> full fan-in column
        nn.Conv1d      -> kernel-sparse column (actual kernel/padding/stride)
        nn.Sequential  -> recursed
        nn.ModuleList  -> recursed (each element processed in order)
        Activations / BN / Dropout -> skipped (no new column)

    Returns:
        layers_info : list[LayerInfo]  (includes the input stub at index 0)
        subtitle    : str
    """
    # ── Model-defined override (preferred for custom architectures) ────
    if hasattr(model, "graph_layers"):
        infos = model.graph_layers()
        return infos, _subtitle_from_layers(infos)

    # ── Generic introspection ──────────────────────────────────────────
    layers_info: list[LayerInfo] = [
        LayerInfo(n_nodes=1, label="in", connectivity=full_connectivity)
    ]

    conv_count = 0
    fc_count   = 0

    def _process_module(mod: nn.Module) -> None:
        nonlocal conv_count, fc_count
        if isinstance(mod, nn.Conv1d):
            conv_count += 1
            k = mod.kernel_size[0]
            p = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
            s = mod.stride[0]  if isinstance(mod.stride,  tuple) else mod.stride
            label = f"conv{conv_count}\n{mod.out_channels}n\nk={k} p={p} s={s}"
            if mod.in_channels == 1 or mod.out_channels == 1:
                conn = full_connectivity
            else:
                conn = conv1d_connectivity(k, p, s)
            layers_info.append(LayerInfo(n_nodes=mod.out_channels,
                                         label=label, connectivity=conn))
        elif isinstance(mod, nn.Linear):
            fc_count += 1
            label = f"fc{fc_count}\n{mod.out_features}n"
            layers_info.append(LayerInfo(n_nodes=mod.out_features,
                                         label=label,
                                         connectivity=full_connectivity))
        elif isinstance(mod, (nn.Sequential, nn.ModuleList)):
            for child in mod.children():
                _process_module(child)
        # Activations, BN, Dropout, custom layers without Linear/Conv1d → skip

    # ── Walk top-level named sub-modules in declaration order ─────────
    for name, child in model.named_children():
        if isinstance(child, nn.Conv1d):
            # bare Conv1d (e.g. CNN.head)
            k = child.kernel_size[0]
            p = child.padding[0] if isinstance(child.padding, tuple) else child.padding
            s = child.stride[0]  if isinstance(child.stride,  tuple) else child.stride
            label = f"out\n1n\nk={k}"
            conn  = full_connectivity if (child.in_channels == 1 or child.out_channels == 1) \
                    else conv1d_connectivity(k, p, s)
            layers_info.append(LayerInfo(n_nodes=child.out_channels,
                                         label=label, connectivity=conn))
        else:
            _process_module(child)

    # ── Ensure output stub exists ──────────────────────────────────────
    if layers_info[-1].label != "out" and layers_info[-1].n_nodes != 1:
        layers_info.append(LayerInfo(
            n_nodes=1, label="out", connectivity=full_connectivity
        ))

    subtitle = _subtitle_from_layers(layers_info)
    return layers_info, subtitle


# ── draw_graph ────────────────────────────────────────────────────────
def draw_graph(ax: plt.Axes, model: nn.Module, ellipsize_after: int = 8) -> None:
    """
    Draw the network architecture on *ax*.

    Architecture-agnostic: works for FCNN, CNN (arbitrary kernel/padding/stride),
    Highway networks, DenseNets, or any hybrid — as long as the model is composed
    of nn.Linear and/or nn.Conv1d layers arranged in nn.Sequential sub-modules
    (the standard PyTorch idiom).

    Connectivity is derived directly from each layer's actual attributes
    (kernel_size, padding, stride) rather than hard-coded assumptions.

    Layers with more nodes than *ellipsize_after* show the first
    (ellipsize_after-1) nodes plus a grey ellipsis marker.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")

    layers_info, subtitle = introspect_model(model)
    weights = extract_layer_weights(model)
    layer_weight_idx = 0
    max_weight = max(
        np.max(np.abs(w)) for w in weights
    ) if weights else 1.0
        
    n_cols = len(layers_info)
    x_pos  = np.linspace(0, (n_cols - 1) * 3, n_cols)

    def n_shown(ch: int) -> int:
        return min(ch, ellipsize_after)

    def y_coords(ch: int) -> np.ndarray:
        n = n_shown(ch)
        return np.linspace(-(n - 1) / 2, (n - 1) / 2, n)

    # ── Edges (primary connections) ───────────────────────────────────
    for li in range(1, n_cols):
        info_in  = layers_info[li - 1]
        info_out = layers_info[li]
        ys_in    = y_coords(info_in.n_nodes)
        ys_out   = y_coords(info_out.n_nodes)
        n_in     = n_shown(info_in.n_nodes)
        n_out    = n_shown(info_out.n_nodes)
        
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
                            w = weight_matrix[j, src]
                            alpha = abs(w)/max_weight
                            color = "#d62728" if w < 0 else "#1f77b4"
                        except Exception:
                            pass
                    ax.plot(
                    [x_pos[li - 1], x_pos[li]],
                    [ys_in[src], ys_out[j]],
                    color=color,
                    lw=1,
                    alpha=alpha,
                    zorder=1)

    # ── Edges (skip / residual connections) ───────────────────────────
    for li, info_out in enumerate(layers_info):
        for layer_offset, skip_srcs in info_out.extra_srcs:
            src_li = li + layer_offset          # layer_offset is negative
            if src_li < 0:
                continue
            info_skip = layers_info[src_li]
            ys_skip   = y_coords(info_skip.n_nodes)
            ys_out    = y_coords(info_out.n_nodes)
            n_skip    = n_shown(info_skip.n_nodes)
            for j, src in enumerate(skip_srcs):
                if 0 <= src < n_skip and j < n_shown(info_out.n_nodes):
                    ax.annotate(
                        "", xy=(x_pos[li], ys_out[j]),
                        xytext=(x_pos[src_li], ys_skip[src]),
                        arrowprops=dict(
                            arrowstyle="->", color="#999999",
                            lw=1, alpha=0.8, connectionstyle="arc3,rad=0.35",
                        ),
                        zorder=1,
                    )

    # ── Nodes ─────────────────────────────────────────────────────────
    node_r = 0.22
    dot_r  = 0.06
    for li, info in enumerate(layers_info):
        ys            = y_coords(info.n_nodes)
        is_ellipsized = info.n_nodes > ellipsize_after
        for i, y in enumerate(ys):
            is_dots = is_ellipsized and i == len(ys) - 1
            if is_dots:
                spacing = node_r * 0.9
                for dy in (-spacing, 0, spacing):
                    ax.add_patch(plt.Circle(
                        (x_pos[li], y + dy), dot_r,
                        color="#444444", zorder=2, linewidth=0,
                    ))
            else:
                ax.add_patch(plt.Circle(
                    (x_pos[li], y), node_r,
                    facecolor="#5B9BD5", zorder=2,
                    linewidth=0.8, edgecolor="white",
                ))

    # ── Column labels ─────────────────────────────────────────────────
    max_shown = max(n_shown(info.n_nodes) for info in layers_info)
    y_bot     = -(max_shown / 2) - 0.9

    for li, info in enumerate(layers_info):
        ax.text(x_pos[li], y_bot, info.label,
                ha="center", va="top", fontsize=7, color="#444444")

    ax.set_title(f"Network graph  ({subtitle})", fontsize=8, pad=4)
    margin = 0.8
    ax.set_xlim(x_pos[0] - margin, x_pos[-1] + margin)
    ax.set_ylim(y_bot - 0.2, max_shown / 2 + margin)


# ── Animation factory ─────────────────────────────────────────────────
def make_animation(
    model:      nn.Module,
    snapshots:  list,
    mse_losses: list,
    ce_losses:  list,
    x_np:       np.ndarray,
    y_np:       np.ndarray,
    epochs:     int,
    title:      str,
    pred_color: str,
    mse_color:  str,
    ce_color:   str,
) -> tuple:
    """
    Build a 2x2 animated figure:
        [top-left]     curve fit (animated)
        [top-right]    network graph (static)
        [bottom-left]  MSE loss curve (animated)
        [bottom-right] cross-entropy loss curve (animated)

    Returns (fig, ani).
    """
    epoch_vals = np.arange(1, epochs + 1)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax_fit   = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_mse   = fig.add_subplot(gs[1, 0])
    ax_ce    = fig.add_subplot(gs[1, 1])

    # ── Top-left: curve fit ───────────────────────────────────────────
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa", label="Data", zorder=1)
    ax_fit.set_xlim(np.min(x_np) * 1.05, np.max(x_np) * 1.05)
    ax_fit.set_ylim(np.min(y_np) * 1.2, np.max(y_np) * 1.2)
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

    # ── Bottom-right: cross-entropy loss ──────────────────────────────
    ax_ce.set_xlim(1, epochs)
    ax_ce.set_ylim(max(min(ce_losses) * 0.5, 1e-9), max(ce_losses) * 1.1)
    ax_ce.set_yscale("log")
    ax_ce.set_xlabel("Epoch")
    ax_ce.set_ylabel("Cross-Entropy Loss")
    ax_ce.set_title("Cross-Entropy Loss")

    # ── Dynamic artists ───────────────────────────────────────────────
    (line_pred,) = ax_fit.plot([], [], lw=2, color=pred_color, zorder=2, label="Prediction")
    fit_title    = ax_fit.set_title("")
    ax_fit.legend(loc="upper right", fontsize=8)

    (line_mse,) = ax_mse.plot([], [], lw=1.5, color=mse_color)
    dot_mse,    = ax_mse.plot([], [], "o",    color=pred_color, ms=6, zorder=3)

    (line_ce,)  = ax_ce.plot([], [], lw=1.5, color=ce_color)
    dot_ce,     = ax_ce.plot([], [], "o",    color=pred_color, ms=6, zorder=3)

    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.35)

    def update(frame_idx: int):
        epoch, mse_val, ce_val, y_pred = snapshots[frame_idx]
        mask = epoch_vals <= epoch

        line_pred.set_data(x_np, y_pred)
        fit_title.set_text(f"Epoch {epoch:>4d}")

        line_mse.set_data(epoch_vals[mask], np.array(mse_losses)[mask])
        dot_mse.set_data([epoch], [mse_val])

        line_ce.set_data(epoch_vals[mask], np.array(ce_losses)[mask])
        dot_ce.set_data([epoch], [ce_val])

        return line_pred, fit_title, line_mse, dot_mse, line_ce, dot_ce

    ani = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=80,
        blit=True,
        repeat=False,
    )

    return fig, ani