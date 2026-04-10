import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from models import ConvNet as ConvNet

# ── graph helper ──────────────────────────────────────────────────────
def draw_graph(ax: plt.Axes, model: nn.Module, ellipsize_after: int = 8) -> None:
    """
    Draw the network architecture on ax using a custom matplotlib renderer.
 
    Works for both DenseNet (fully-connected) and ConvNet (kernel-sparse):
      - DenseNet  : every node in layer N connects to every node in N+1.
      - ConvNet   : each output node j connects only to input nodes
                    j-1, j, j+1  (kernel=3, padding=1 receptive field).
 
    Layers with more nodes than ellipsize_after display the first
    (ellipsize_after-1) nodes plus one grey ellipsis node.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
 
    is_conv = isinstance(model, ConvNet)
 
    # ── Collect (size, kernel) per layer ──────────────────────────────
    # Each entry: (n_nodes, kernel_radius)
    # kernel_radius=0  -> fully connected  (Linear / input)
    # kernel_radius=1  -> kernel=3         (Conv1d with padding=1)
    layers_info: list[tuple[int, int]] = []
 
    if is_conv:
        layers_info.append((1, 0))                        # input channel
        for layer in model.conv.children():
            if isinstance(layer, nn.Conv1d):
                layers_info.append((layer.out_channels, layer.kernel_size[0] // 2))
        layers_info.append((1, 0))                        # head output
        subtitle = "kernel=3, stride=1, padding=1"
    else:
        layers_info.append((1, 0))                        # input feature
        for layer in model.net.children():
            if isinstance(layer, nn.Linear):
                layers_info.append((layer.out_features, 0))
        subtitle = "fully connected"
 
    # ── Layout ────────────────────────────────────────────────────────
    n_cols = len(layers_info)
    x_pos  = np.linspace(0, (n_cols - 1) * 2.2, n_cols)
 
    def n_shown(ch: int) -> int:
        return min(ch, ellipsize_after)
 
    def y_coords(ch: int) -> np.ndarray:
        n = n_shown(ch)
        return np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
 
    # ── Edges ─────────────────────────────────────────────────────────
    for li in range(1, n_cols):
        ch_in,  _       = layers_info[li - 1]
        ch_out, half_k  = layers_info[li]
        ys_in  = y_coords(ch_in)
        ys_out = y_coords(ch_out)
        n_in   = n_shown(ch_in)
        n_out  = n_shown(ch_out)
 
        for j in range(n_out):
            if half_k == 0 or ch_in == 1:
                # Fully connected: single input node always fans out to all outputs
                src_range = range(n_in)
            else:
                # Kernel-sparse: j connects to j-half_k … j+half_k
                src_range = range(
                    max(0, j - half_k),
                    min(ch_in, j + half_k + 1),
                )
                src_range = [s for s in src_range if s < n_in]
 
            for src in src_range:
                ax.plot(
                    [x_pos[li - 1], x_pos[li]],
                    [ys_in[src], ys_out[j]],
                    color="#999999", lw=0.6, alpha=0.45, zorder=1,
                )
 
    # ── Nodes ─────────────────────────────────────────────────────────
    node_r = 0.18
    dot_r  = 0.06   # radius of each small ellipsis dot
    for li, (ch, _) in enumerate(layers_info):
        ys            = y_coords(ch)
        is_ellipsized = ch > ellipsize_after
        for i, y in enumerate(ys):
            is_dots = is_ellipsized and i == len(ys) - 1
            if is_dots:
                # Draw three small black dots stacked vertically
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
    max_shown = max(n_shown(ch) for ch, _ in layers_info)
    y_bot     = -(max_shown / 2) - 0.9
 
    if is_conv:
        labels = ["in"] + [
            f"conv{i+1}\n{ch}ch" for i, (ch, _) in enumerate(layers_info[1:-1])
        ] + ["out"]
    else:
        labels = ["in"] + [
            f"fc{i+1}\n{ch}" for i, (ch, _) in enumerate(layers_info[1:-1])
        ] + ["out"]
 
    for li, label in enumerate(labels):
        ax.text(x_pos[li], y_bot, label, ha="center", va="top",
                fontsize=7, color="#444444")
 
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
        [top-right]    network graph via visualtorch (static)
        [bottom-left]  MSE loss curve (animated)
        [bottom-right] cross-entropy loss curve (animated)
    
    Returns (fig, ani).
    """

    epoch_vals = np.arange(1, epochs + 1)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax_fit   = fig.add_subplot(gs[0, 0])   # top-left
    ax_graph = fig.add_subplot(gs[0, 1])   # top-right
    ax_mse   = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_ce    = fig.add_subplot(gs[1, 1])   # bottom-right

    # ── Top-left: curve fit ───────────────────────────────────────────
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa", label="Data", zorder=1)
    ax_fit.set_xlim(np.min(x_np)*1.2, np.max(x_np)*1.2)
    ax_fit.set_ylim(np.min(y_np)*1.2, np.max(y_np)*1.2)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
    ax_fit.set_title("Curve fit")
    
    # ── Top-right: network graph (static, rendered once) ─────────────
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