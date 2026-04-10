import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.models import FCNN
from utils.models import CNN
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Shared config ─────────────────────────────────────────────────
    H_N       = 3               # number of hidden / conv layers
    N_N       = 15              # nodes per dense layer / filters per conv layer
    FUNC      = nn.Tanh()       # activation: nn.Tanh() | nn.ReLU() | nn.Sigmoid | nn.Softplus | nn.Softshrink | nn.Softsign | nn.Mish | etc.
    EPOCHS    = 2000
    LR        = 1e-2
    LOG_EVERY = 50              # snapshot + print interval
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    INPUT_N   = 200
    NOISE_STD = 0.2
    # ──────────────────────────────────────────────────────────────────

    # ── Synthetic data (non-uniform Gaussian noise) ───────────────────
    x_np      = np.linspace(0, 5, INPUT_N).astype(np.float32)
    y         = test_func(x_np)
    y_np      = (y + np.random.normal(y, scale=NOISE_STD)).astype(np.float32)

    x = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y = torch.tensor(y_np).unsqueeze(1)   # (200, 1)

    act_name = FUNC.__class__.__name__

    # ── Train FCNN ────────────────────────────────────────────────
    dense_model = FCNN(h_n=H_N, n_n=N_N, func=FUNC)
    dense_snaps, dense_mse, dense_ce = train_model(
        dense_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"FCNN  |  layers={H_N}  nodes={N_N}  act={act_name}",
    )

    # ── Train CNN ─────────────────────────────────────────────────
    conv_model = CNN(h_n=H_N, n_n=N_N, func=FUNC, kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    conv_snaps, conv_mse, conv_ce = train_model(
        conv_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"CNN   |  layers={H_N}  filters={N_N}  act={act_name}",
    )

    # ── Animate & save FCNN GIF ───────────────────────────────────
    fig_dense, ani_dense = make_animation(
        model        = dense_model,
        snapshots    = dense_snaps,
        mse_losses   = dense_mse,
        ce_losses    = dense_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"Fully Connected NN — {H_N} hidden layers × {N_N} nodes  [{act_name}]",
        pred_color   = "#e05c2e",
        mse_color    = "#2e7de0",
        ce_color     = "#7c3aed",
    )
    print("\nSaving FCNN_training.gif …")
    ani_dense.save("FCNN_training.gif", writer="pillow", fps=15)
    print("Saved  →  FCNN_training.gif")

    # ── Animate & save CNN GIF ────────────────────────────────────
    fig_conv, ani_conv = make_animation(
        model        = conv_model,
        snapshots    = conv_snaps,
        mse_losses   = conv_mse,
        ce_losses    = conv_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"Convolutional NN — {H_N} conv layers × {N_N} filters  [{act_name}]",
        pred_color   = "#e05c2e",
        mse_color    = "#2e7de0",
        ce_color     = "#7c3aed",
    )
    print("Saving CNN_training.gif …")
    ani_conv.save("CNN_training.gif", writer="pillow", fps=15)
    print("Saved  →  CNN_training.gif")
    
    for fig in (fig_dense, fig_conv):
        fig.set_size_inches(9, 6)
    plt.show()