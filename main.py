import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from utils.models import FCNN
from utils.models import CNN
from utils.models import DenseResNet
from utils.models import ConvResNet
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Shared config ─────────────────────────────────────────────────
    H_N       = 5 # number of hidden / conv layers
    N_N       = 8 # nodes per dense layer / filters per conv layer
    FUNC      = nn.Mish() # activation: nn.Tanh | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Softshrink | nn.Softsign | nn.Mish | etc.
    EPOCHS    = 1000
    LR        = 1e-2
    LOG_EVERY = 10 # snapshot + print interval
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    CONNECT   = 3
    INPUT_N   = 200
    NOISE_STD = 0.1
    SHOW      = True # whether to show the figures or not
    FILE_TYPE = 'gif' # save file type: gif | png | jpeg
    SAVE      = "saves"
    # ──────────────────────────────────────────────────────────────────

    # ── Synthetic data (non-uniform Gaussian noise) ───────────────────
    x_np      = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y         = test_func(x_np)
    y_np      = (y + np.random.normal(size=y.shape, scale=NOISE_STD)).astype(np.float32)

    x = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y = torch.tensor(y_np).unsqueeze(1)   # (200, 1)

    act_name = FUNC.__class__.__name__
    if not os.path.isdir(SAVE):
        os.mkdir(SAVE)
    SAVE = os.path.join(SAVE, act_name)
    # ── Train FCNN ────────────────────────────────────────────────
    dense_model = FCNN(h_n=H_N, n_n=N_N, func=FUNC)
    dense_snaps, dense_mse, dense_bce = train_model(
        dense_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"Fully Connected NN | {H_N} hidden layers × {N_N} nodes  [{act_name}]",
    )

    # ── Train CNN ─────────────────────────────────────────────────
    conv_model = CNN(h_n=H_N, n_n=N_N, func=FUNC, kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    conv_snaps, conv_mse, conv_bce = train_model(
        conv_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"Convolutional NN | {H_N} conv layers × {N_N} filters  [{act_name}]",
    )

    # ── Train DenseResNet ─────────────────────────────────────────────
    dense_res_model = DenseResNet(h_n=H_N, n_n=N_N, func=FUNC)
    dense_res_snaps, dense_res_mse, dense_res_ce = train_model(
        dense_res_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"Dense ResNet | {H_N} layers × {N_N} nodes  [{act_name}]",
    )

    # ── Train ConvResNet ──────────────────────────────────────────────
    conv_res_model = ConvResNet(
    h_n=H_N, n_n=N_N, connect=CONNECT, func=FUNC, kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    conv_res_snaps, conv_res_mse, conv_res_ce = train_model(
        conv_res_model, x, y, EPOCHS, LR, LOG_EVERY,
        label=f"Conv ResNet | {H_N} conv layers × {N_N} filters  connect={CONNECT}  [{act_name}]",
    )

    # ── Animate & save FCNN ───────────────────────────────────────────
    fig_dense, ani_dense = make_animation(
        model = dense_model,
        snapshots = dense_snaps,
        mse_losses = dense_mse,
        bce_losses = dense_bce,
        x_np = x_np,
        y_np = y_np,
        epochs = EPOCHS,
        title = f"Fully Connected NN | {H_N} hidden layers × {N_N} nodes  [{act_name}]",
        pred_color = "#e05c2e",
        mse_color = "#2e7de0",
        bce_color = "#7c3aed",
        file_type = FILE_TYPE,
        savepath = SAVE
    )

    # ── Animate & save CNN ────────────────────────────────────────────
    fig_conv, ani_conv = make_animation(
        model = conv_model,
        snapshots = conv_snaps,
        mse_losses = conv_mse,
        bce_losses = conv_bce,
        x_np = x_np,
        y_np = y_np,
        epochs = EPOCHS,
        title = f"Convolutional NN | {H_N} conv layers × {N_N} filters  [{act_name}]",
        pred_color = "#e05c2e",
        mse_color = "#2e7de0",
        bce_color = "#7c3aed",
        file_type = FILE_TYPE,
        savepath = SAVE
    )

    # ── Animate & save DenseResNet ─────────────────────────────────────
    fig_dense_res, ani_dense_res = make_animation(
        model = dense_res_model,
        snapshots = dense_res_snaps,
        mse_losses = dense_res_mse,
        bce_losses = dense_res_ce,
        x_np = x_np,
        y_np = y_np,
        epochs = EPOCHS,
        title = f"Dense ResNet | {H_N} layers × {N_N} nodes  [{act_name}]",
        pred_color = "#e05c2e",
        mse_color = "#2e7de0",
        bce_color = "#7c3aed",
        file_type = FILE_TYPE,
        savepath = SAVE
    )

    # ── Animate & save ConvResNet ──────────────────────────────────────
    fig_conv_res, ani_conv_res = make_animation(
        model = conv_res_model,
        snapshots = conv_res_snaps,
        mse_losses = conv_res_mse,
        bce_losses = conv_res_ce,
        x_np = x_np,
        y_np = y_np,
        epochs = EPOCHS,
        title = f"Conv ResNet | {H_N} conv layers × {N_N} filters  connect={CONNECT}  [{act_name}]",
        pred_color = "#e05c2e",
        mse_color = "#2e7de0",
        bce_color = "#7c3aed",
        file_type = FILE_TYPE,
        savepath = SAVE
    )
    
    if SHOW:
        for fig in (fig_dense, fig_conv, fig_dense_res, fig_conv_res):
            fig.set_size_inches(7, 5)
        plt.show()