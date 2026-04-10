import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.models import FCNN
from utils.models import CNN
from utils.models import DenseHNN
from utils.models import ConvHNN
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Shared config ─────────────────────────────────────────────────
    H_N       = 5               # number of hidden / conv layers
    N_N       = 15              # nodes per dense layer / filters per conv layer
    FUNC      = nn.Tanh()       # activation: nn.Tanh() | nn.ReLU() | nn.Sigmoid | nn.Softplus | nn.Softshrink | nn.Softsign | nn.Mish | etc.
    EPOCHS    = 500
    DROPOUT   = 0.2
    LR        = 1e-2
    LOG_EVERY = 10              # snapshot + print interval
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    INPUT_N   = 200
    NOISE_STD = 0.1
    # ──────────────────────────────────────────────────────────────────

    # ── Synthetic data (non-uniform Gaussian noise) ───────────────────
    x_np      = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y         = test_func(x_np)
    y_np      = (y + np.random.normal(size=y.shape, scale=NOISE_STD)).astype(np.float32)

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

    # ── Train Dense HNN ───────────────────────────────────────────────
    dense_hnn_model = DenseHNN(h_n=H_N, n_n=N_N, func=FUNC)
    dense_hnn_snaps, dense_hnn_mse, dense_hnn_ce = train_model(
        dense_hnn_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"DenseHNN  |  layers={H_N}  nodes={N_N}  act={act_name}",
    )

    # ── Train Conv HNN ────────────────────────────────────────────────
    conv_hnn_model = ConvHNN(h_n=H_N, n_n=N_N, func=FUNC, kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    conv_hnn_snaps, conv_hnn_mse, conv_hnn_ce = train_model(
        conv_hnn_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"ConvHNN   |  layers={H_N}  nodes={N_N}  k={K_SIZE}  p={PADDING}  s={STRIDE}  act={act_name}",
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
    print("\nSaving FCNN_training.gif … ", end="")
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
    print("Saving CNN_training.gif … ", end="")
    ani_conv.save("CNN_training.gif", writer="pillow", fps=15)
    print("Saved  →  CNN_training.gif")

    # ── Animate & save Dense HNN GIF ─────────────────────────────────
    fig_dense_hnn, ani_dense_hnn = make_animation(
        model        = dense_hnn_model,
        snapshots    = dense_hnn_snaps,
        mse_losses   = dense_hnn_mse,
        ce_losses    = dense_hnn_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"Dense Highway NN — {H_N} layers × {N_N} nodes  [{act_name}]",
        pred_color   = "#e05c2e",
        mse_color    = "#2e7de0",
        ce_color     = "#7c3aed",
    )
    print("Saving DenseHNN_training.gif … ", end="")
    ani_dense_hnn.save("DenseHNN_training.gif", writer="pillow", fps=15)
    print("Saved  →  DenseHNN_training.gif")

    # ── Animate & save Conv HNN GIF ───────────────────────────────────
    fig_conv_hnn, ani_conv_hnn = make_animation(
        model        = conv_hnn_model,
        snapshots    = conv_hnn_snaps,
        mse_losses   = conv_hnn_mse,
        ce_losses    = conv_hnn_ce,
        x_np         = x_np,
        y_np         = y_np,
        epochs       = EPOCHS,
        title        = f"Conv Highway NN — {H_N} layers × {N_N} nodes  k={K_SIZE} p={PADDING} s={STRIDE}  [{act_name}]",
        pred_color   = "#e05c2e",
        mse_color    = "#2e7de0",
        ce_color     = "#7c3aed",
    )
    print("Saving ConvHNN_training.gif … ", end="")
    ani_conv_hnn.save("ConvHNN_training.gif", writer="pillow", fps=15)
    print("Saved  →  ConvHNN_training.gif")
    
    for fig in (fig_dense, fig_conv, fig_dense_hnn, fig_conv_hnn):
        fig.set_size_inches(7, 5)
    plt.show()