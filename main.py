import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataclasses import replace

from utils.parameters import ModelParams, apply_seed, FUNC_DICT, LOSS_FUNC_DICT
from utils.models import FCNN, CNN, DenseResNet, ConvResNet, CustomNet
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

if __name__ == "__main__":

    # Configs -----------------------------------------------------------------
    H_N       = 5 # number of hidden / conv layers
    N_N       = 8 # nodes per dense layer / filters per conv layer
    FUNC      = FUNC_DICT[17] # FUNC_DICT[17] is nn.Mish()
    LOSS1     = LOSS_FUNC_DICT[1]
    LOSS2     = LOSS_FUNC_DICT[4]
    LR        = 1e-2
    EPOCHS    = 1000
    LOG_EVERY = 10
    GRAD_CLIP = 100
    SEED      = 1
    SHUFFLE   = True
    DEVICE    = "cpu"
    VERBOSE   = False
    
    K_SIZE    = 3
    PADDING   = 1
    STRIDE    = 1
    CONNECT   = 3
    
    INPUT_N   = 200
    NOISE_STD = 0.1
    SHOW      = True # whether to show the figures or not
    FILE_TYPE = "gif" # save file type: gif | png | jpeg
    SAVE      = "saves"
    
    PRED_COLOR  = "#e05c2e"
    LOSS_COLOR  = "#2e7de0"
    LOSS2_COLOR = "#7c3aed"

    # Synthetic data (non-uniform Gaussian noise) -----------------------------
    apply_seed(SEED) # to apply seeds for numpy random functions
    x_np = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y = test_func(x_np)
    y_np = (y + np.random.normal(size=y.shape, scale=NOISE_STD)).astype(np.float32)

    x_t = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y_t = torch.tensor(y_np).unsqueeze(1)   # (200, 1)

    os.makedirs(SAVE, exist_ok=True)
    SAVE = os.path.join(SAVE, FUNC.__class__.__name__)
    params = ModelParams(
        name    = "",
        layer_sizes = [1] + [N_N] * H_N + [1],
        activation_functions = [FUNC] * H_N,
        learning_rate = LR,
        max_epoch = EPOCHS,
        print_each = LOG_EVERY,
        gradient_clip = GRAD_CLIP,
        seed = SEED,
        shuffle = SHUFFLE,
        loss_function = LOSS1,
        loss_function2 = LOSS2,
        device = DEVICE,
        verbose = VERBOSE,
    )
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
    
    for src_layer in range(H_N+1):
        src_size = layer_sizes[src_layer]
        tgt_layer = src_layer + 1
        if src_layer == 0 or src_layer == H_N:
            for src_node in range(layer_sizes[src_layer]):
                nodes.extend([src_layer, src_node, tgt_layer, tgt_node] for tgt_node in range(layer_sizes[tgt_layer]))
        else:
            for src_node in range(layer_sizes[src_layer]):
                target_nodes = [(src_node+1)%N_N, (src_node+N_N//2)%N_N]
                nodes.extend([src_layer, src_node, tgt_layer, tgt_node] for tgt_node in target_nodes)
            if src_layer == 1 and len(layer_sizes) > 4:
                nodes.extend([src_layer, src_node, H_N, src_node] for src_node in range(layer_sizes[src_layer]))
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