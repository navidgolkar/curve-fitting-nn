import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

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

        layers = [nn.Linear(1, n_n), func]
        for _ in range(h_n - 1):
            layers += [nn.Linear(n_n, n_n), func]
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
    def __init__(self, h_n: int, n_n: int, func: nn.Module):
        super().__init__()
        self.h_n  = h_n
        self.n_n  = n_n
        self.func = func

        conv_layers: list = []
        in_ch = 1
        for _ in range(h_n):
            conv_layers += [
                nn.Conv1d(in_ch, n_n, kernel_size=3, padding=1),  # same-padding keeps length
                func,
            ]
            in_ch = n_n
        self.conv = nn.Sequential(*conv_layers)

        # Pointwise head: project n_n channels -> 1 at each position
        self.head = nn.Conv1d(n_n, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (N, 1)
        # Conv1d expects (batch, channels, length) -> reshape to (1, 1, N)
        out = x.T.unsqueeze(0)   # (1, 1, N)
        out = self.conv(out)     # (1, n_n, N)
        out = self.head(out)     # (1, 1, N)
        return out.squeeze(0).T  # (N, 1)


# ── Shared training loop ──────────────────────────────────────────────
def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    log_every: int,
    label: str,
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
    """Build an animated 3-panel figure and return (fig, ani)."""

    epoch_vals = np.arange(1, epochs + 1)

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax_fit = fig.add_subplot(gs[0])
    ax_mse = fig.add_subplot(gs[1])
    ax_ce  = fig.add_subplot(gs[2])

    # Fit panel
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa", label="Data", zorder=1)
    ax_fit.set_xlim(-3.2, 3.2)
    ax_fit.set_ylim(-1.8, 1.8)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")

    # MSE panel
    ax_mse.set_xlim(1, epochs)
    ax_mse.set_ylim(max(min(mse_losses) * 0.5, 1e-9), max(mse_losses) * 1.1)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE Loss")
    ax_mse.set_title("MSE Loss")

    # CE panel
    ax_ce.set_xlim(1, epochs)
    ax_ce.set_ylim(max(min(ce_losses) * 0.5, 1e-9), max(ce_losses) * 1.1)
    ax_ce.set_yscale("log")
    ax_ce.set_xlabel("Epoch")
    ax_ce.set_ylabel("Cross-Entropy Loss")
    ax_ce.set_title("Cross-Entropy Loss")

    # Dynamic artists
    (line_pred,) = ax_fit.plot([], [], lw=2, color=pred_color, zorder=2, label="Prediction")
    fit_title    = ax_fit.set_title("")
    ax_fit.legend(loc="upper right", fontsize=8)

    (line_mse,) = ax_mse.plot([], [], lw=1.5, color=mse_color)
    dot_mse,    = ax_mse.plot([], [], "o",    color=pred_color, ms=6, zorder=3)

    (line_ce,)  = ax_ce.plot([], [], lw=1.5, color=ce_color)
    dot_ce,     = ax_ce.plot([], [], "o",    color=pred_color, ms=6, zorder=3)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(top=0.88, wspace=0.4)

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
    # ──────────────────────────────────────────────────────────────────

    # ── Synthetic data (non-uniform Gaussian noise) ───────────────────
    x_np      = np.linspace(-3, 3, 200).astype(np.float32)
    noise_std = 0.05 + 0.3 * np.exp(-0.5 * ((x_np - 0.5) / 1.2) ** 2)
    y_np      = (np.sin(x_np) + noise_std * np.random.randn(*x_np.shape)).astype(np.float32)

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
    conv_model = ConvNet(h_n=H_N, n_n=N_N, func=FUNC)
    conv_snaps, conv_mse, conv_ce = train_model(
        conv_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"ConvNet   |  layers={H_N}  filters={N_N}  act={act_name}",
    )

    # ── Animate & save DenseNet GIF ───────────────────────────────────
    fig_dense, ani_dense = make_animation(
        snapshots  = dense_snaps,
        mse_losses = dense_mse,
        ce_losses  = dense_ce,
        x_np       = x_np,
        y_np       = y_np,
        epochs     = EPOCHS,
        title      = f"DenseNet — {H_N} hidden layers × {N_N} nodes  [{act_name}]",
        pred_color = "#e05c2e",
        mse_color  = "#2e7de0",
        ce_color   = "#7c3aed",
    )
    print("\nSaving densenet_training.gif …")
    ani_dense.save("densenet_training.gif", writer="pillow", fps=15)
    print("Saved  →  densenet_training.gif")

    # ── Animate & save ConvNet GIF ────────────────────────────────────
    fig_conv, ani_conv = make_animation(
        snapshots  = conv_snaps,
        mse_losses = conv_mse,
        ce_losses  = conv_ce,
        x_np       = x_np,
        y_np       = y_np,
        epochs     = EPOCHS,
        title      = f"ConvNet — {H_N} conv layers × {N_N} filters  [{act_name}]",
        pred_color = "#0d9e6e",
        mse_color  = "#e0a02e",
        ce_color   = "#e05c2e",
    )
    print("Saving convnet_training.gif …")
    ani_conv.save("convnet_training.gif", writer="pillow", fps=15)
    print("Saved  →  convnet_training.gif")

    plt.show()