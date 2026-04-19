import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from dataclasses import replace

from utils.parameters import ModelParams, apply_seed
from utils.models import FCNN, CNN, DenseResNet, ConvResNet, CustomNet
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

if __name__ == "__main__":

    # Shared config -----------------------------------------------------------
    H_N       = 4 # number of hidden / conv layers
    N_N       = 5 # nodes per dense layer / filters per conv layer
    FUNC      = nn.Mish() # activation: nn.Tanh | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Softshrink | nn.Softsign | nn.Mish | etc.
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    CONNECT   = 3
    
    params = ModelParams(
        name    = "",
        layer_sizes = [1] + [N_N] * H_N + [1],
        activation_functions = [FUNC] * H_N,
        learning_rate = 1e-2,
        max_epoch = 1000,
        print_each = 10,
        gradient_clip = 100,
        seed = 1,
        shuffle = True,
        loss_function = nn.MSELoss(),
        loss_function2 = nn.BCEWithLogitsLoss(),
        device = "cpu",
        verbose = False,
    )
    
    INPUT_N   = 200
    NOISE_STD = 0.1
    SHOW      = False # whether to show the figures or not
    FILE_TYPE = "png" # save file type: gif | png | jpeg
    SAVE      = "saves"
    
    PRED_COLOR  = "#e05c2e"
    LOSS_COLOR  = "#2e7de0"
    LOSS2_COLOR = "#7c3aed"

    # Synthetic data (non-uniform Gaussian noise) -----------------------------
    apply_seed(params.seed)
    x_np = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y = test_func(x_np)
    y_np = (y + np.random.normal(size=y.shape, scale=NOISE_STD)).astype(np.float32)

    x_t = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y_t = torch.tensor(y_np).unsqueeze(1)   # (200, 1)

    os.makedirs(SAVE, exist_ok=True)
    SAVE = os.path.join(SAVE, FUNC.__class__.__name__)
    
    results = []
    
    # ======================================================================= #
    # FCNN                                                                    #
    # ======================================================================= #
    model = FCNN(replace(params, name="FCNN"))
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # CNN                                                                     #
    # ======================================================================= #
    model = CNN(replace(params, name="CNN"), kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # DenseResNet                                                             #
    # ======================================================================= #
    model = DenseResNet(replace(params, name="DenseResNet"))
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # ConvResNet                                                              #
    # ======================================================================= #
    model = ConvResNet(replace(params, name="ConvResNet"), kernel_size=K_SIZE, padding=PADDING, stride=STRIDE, connect=CONNECT)
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # CustomNet                                                               #
    # ======================================================================= #
    nodes = []
    layer_sizes  = [1] + [N_N] * H_N + [1]   # [1, 3, 3, 1]
    
    for n_layer in range(H_N+1):
        if n_layer == 0:
            nodes.append([[(n_layer+1, i) for i in range(N_N)]])
        elif n_layer == H_N:
            nodes.append([[(n_layer+1, 0)] for _ in range(N_N)])
        else:
            nodes.append([[(n_layer+j, i) for j in range(1, H_N-n_layer+1)] for i in range(N_N)])
    model = CustomNet(replace(params, name="CustomNet"), nodes)
    results.append((model, *train_model(model, x_t, y_t)))
    
    figs = []
    
    # Animate & save
    for model, snaps, loss, loss2 in results:
        figs.append(make_animation(
            model         = model,
            snapshots     = snaps,
            loss_history  = loss,
            loss2_history = loss2,
            x_np          = x_np,
            y_np          = y_np,
            pred_color    = PRED_COLOR,
            loss_color    = LOSS_COLOR,
            loss2_color   = LOSS2_COLOR,
            file_type     = FILE_TYPE,
            savepath      = SAVE,
        ))
    
    if SHOW:
        for fig, _ in figs:
            fig.set_size_inches(7, 5)
        plt.show()