import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models import DenseNet
from models import ConvNet
from animation import make_animation
from train import train_model

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Shared config ─────────────────────────────────────────────────
    H_N       = 2               # number of hidden / conv layers
    N_N       = 20              # nodes per dense layer / filters per conv layer
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