import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import copy

# ── DenseNet ──────────────────────────────────────────────────────────
class DenseNet(nn.Module):
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


# ── ConvNet ───────────────────────────────────────────────────────────
class ConvNet(nn.Module):
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

# ── Shared training loop ──────────────────────────────────────────────
def train_model(
    model:      nn.Module,
    x:          torch.Tensor,
    y:          torch.Tensor,
    epochs:     int,
    lr:         float,
    log_every:  int,
    label:      str,
) -> tuple:
    """
    Train model with Adam + MSE loss.

    Returns:
        snapshots  : list of (epoch, mse, ce, y_pred_np)
        mse_losses : MSE at every epoch
        ce_losses  : cross-entropy proxy at every epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn    = nn.MSELoss()

    snapshots:  list = []
    mse_losses: list = []
    ce_losses:  list = []

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(model)
    print(f"{'─' * 60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(x)
        mse  = mse_fn(pred, y)
        mse.backward()
        optimizer.step()

        mse_val = mse.item()
        mse_losses.append(mse_val)

        # Cross-entropy proxy: softmax both outputs over the sequence dimension,
        # then compute KL-style CE: -sum(q * log(p))
        with torch.no_grad():
            p      = torch.softmax(pred.squeeze(), dim=0)
            q      = torch.softmax(y.squeeze(),    dim=0)
            ce_val = -torch.sum(q * torch.log(p + 1e-9)).item()
        ce_losses.append(ce_val)

        if epoch % log_every == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x).squeeze().numpy()
            snapshots.append((epoch, mse_val, ce_val, y_pred.copy()))
            print(f"Epoch {epoch:>4d}  |  MSE: {mse_val:.6f}  |  CE: {ce_val:.4f}")

    return snapshots, mse_losses, ce_losses


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
    ax_fit.set_xlim(-3.2, 3.2)
    ax_fit.set_ylim(-1.8, 1.8)
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


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Shared config ─────────────────────────────────────────────────
    H_N       = 3               # number of hidden / conv layers
    N_N       = 64              # nodes per dense layer / filters per conv layer
    FUNC      = nn.Tanh()       # activation: nn.Tanh() | nn.ReLU() | nn.SiLU() | etc.
    EPOCHS    = 2000
    LR        = 1e-3
    LOG_EVERY = 50              # snapshot + print interval
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    # ──────────────────────────────────────────────────────────────────

    # ── Synthetic data (non-uniform Gaussian noise) ───────────────────
    x_np      = np.linspace(-3, 3, 200).astype(np.float32)
    y_np      = (np.sin(x_np) + 0.3 * np.random.randn(*x_np.shape)).astype(np.float32)

    x = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y = torch.tensor(y_np).unsqueeze(1)   # (200, 1)

    act_name = FUNC.__class__.__name__

    # ── Train DenseNet ────────────────────────────────────────────────
    dense_model = DenseNet(h_n=H_N, n_n=N_N, func=FUNC)
    dense_snaps, dense_mse, dense_ce = train_model(
        dense_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"DenseNet  |  layers={H_N}  nodes={N_N}  act={act_name}",
    )

    # ── Train ConvNet ─────────────────────────────────────────────────
    conv_model = ConvNet(h_n=H_N, n_n=N_N, func=FUNC, kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    conv_snaps, conv_mse, conv_ce = train_model(
        conv_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"ConvNet   |  layers={H_N}  filters={N_N}  act={act_name}",
    )

    # ── Animate & save DenseNet GIF ───────────────────────────────────
    fig_dense, ani_dense = make_animation(
        model        = dense_model,
        snapshots    = dense_snaps,
        mse_losses   = dense_mse,
        ce_losses    = dense_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"DenseNet — {H_N} hidden layers × {N_N} nodes  [{act_name}]",
        pred_color   = "#e05c2e",
        mse_color    = "#2e7de0",
        ce_color     = "#7c3aed",
    )
    print("\nSaving densenet_training.gif …")
    ani_dense.save("densenet_training.gif", writer="pillow", fps=15)
    print("Saved  →  densenet_training.gif")

    # ── Animate & save ConvNet GIF ────────────────────────────────────
    fig_conv, ani_conv = make_animation(
        model        = conv_model,
        snapshots    = conv_snaps,
        mse_losses   = conv_mse,
        ce_losses    = conv_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"ConvNet — {H_N} conv layers × {N_N} filters  [{act_name}]",
        pred_color   = "#0d9e6e",
        mse_color    = "#e0a02e",
        ce_color     = "#e05c2e",
    )
    print("Saving convnet_training.gif …")
    ani_conv.save("convnet_training.gif", writer="pillow", fps=15)
    print("Saved  →  convnet_training.gif")
 
    plt.show()