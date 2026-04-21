import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataclasses import replace
import argparse

from utils.parameters import ModelParams, apply_seed, FUNC_DICT, LOSS_FUNC_DICT, OPT_DICT
from utils.models import FCNN, CNN, DenseResNet, ConvResNet, CustomNet
from utils.animation import make_animation
from utils.train import train_model

def test_func(x):
    return 2*np.exp(-x)*(np.sin(5*x)+x*np.cos(5*x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script 4 different configurations of neural networks for curve fitting a 2-dimensional data")

    # Structure Parameters
    parser.add_argument("--hn", required=False, type=int, default=5, help="Number of hidden layers")
    parser.add_argument("--nn", required=False, type=int, default=7, help="number of nodes at each hidden layer (the number of layers is equal at all hidden layers)")
    parser.add_argument("--func", required=False, type=int, default=17, help="1-27: Which activation function to use")
    parser.add_argument("--conv", required=False, type=tuple[int, int, int], default=(3, 1, 1), help="(kernel size, padding, stride) for convolutional neural network")
    parser.add_argument("--connect", required=False, type=int, default=1, help="Number of connections for ConvResNet residual connections")

    # Training Hyper-parameters
    parser.add_argument("--loss1", required=False, type=int, default=2, help="1-9: Which loss unction to use for the training of models")
    parser.add_argument("--opt", required=False, type=int, default=1, help="1-12: Which optimizer to use for training (default is Adam)")
    parser.add_argument("--lr", required=False, type=float, default=1e-2, help="Enter learning rate value")
    parser.add_argument("--grad_clip", required=False, type=float, default=100.0, help="Enter the value at which the gradient should be clipped to prevent explosion")
    parser.add_argument("--tol", required=False, type=float, default=1e-3, help="Enter the tolerance for the network at which to stop training")
    parser.add_argument("--epoch", required=False, type=int, default=3000, help="Number of epochs to run")
    parser.add_argument("--shuffle", required=False, action="store_true", help="Will shuffle input data of models")
    parser.add_argument("--device", required=False, type=str, default="cpu", help="What device to use for pytorch")

    # Outputs and Plots Parameters
    parser.add_argument("--loss2", required=False, type=int, default=2, help="1-9: Which loss unction to use for the second plot (this is not used for training)")
    parser.add_argument("--log", required=False, type=int, default=10, help="the results should should per how many epochs")
    parser.add_argument("--verbose", required=False, action="store_true", help="Whether to show results in console")
    parser.add_argument("--show", required=False, action="store_true", help="Whether to open figure files after running the code")
    parser.add_argument("--file_type", required=False, type=str, default="png", help="What should be the file_type of saved figures (gif, png, jpeg)")
    parser.add_argument("--name", required=False, type=str, default="", help="added string at the end of each file for keeping track at running multiple runs")

    # Input Data Parameters
    parser.add_argument("--in_n", required=False, type=int, default=200, help="Number of input data")
    parser.add_argument("--in_std", required=False, type=float, default=1e-1, help="Standard deviation for input noise")

    # Seed variable influences both training parameters and input data parameters
    parser.add_argument("--seed", required=False, type=int, default=1, help="Seed number for random values")

    # Configs -----------------------------------------------------------------
    args = parser.parse_args()
    H_N       = args.hn
    N_N       = args.nn
    FUNC      = FUNC_DICT[args.func] # FUNC_DICT[17] is nn.Mish()
    OPTIMIZER = OPT_DICT[args.opt]
    LOSS1     = LOSS_FUNC_DICT[args.loss1]
    LOSS2     = LOSS_FUNC_DICT[args.loss2]
    LR        = args.lr
    EPOCHS    = args.epoch
    LOG_EVERY = args.log
    GRAD_CLIP = args.grad_clip
    SEED      = args.seed
    SHUFFLE   = args.shuffle
    DEVICE    = args.device
    VERBOSE   = args.verbose
    NAME      = f"_{args.name}" if args.name != "" else ""
    
    K_SIZE, PADDING, STRIDE = args.conv
    CONNECT   = args.connect
    
    INPUT_N   = args.in_n
    NOISE_STD = args.in_std
    SHOW      = args.show
    FILE_TYPE = args.file_type
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
        optimizer_function = OPTIMIZER,
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
    model = FCNN(replace(params, name=f"FCNN{NAME}"))
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # CNN                                                                     #
    # ======================================================================= #
    model = CNN(replace(params, name=f"CNN{NAME}"), kernel_size=K_SIZE, padding=PADDING, stride=STRIDE)
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # DenseResNet                                                             #
    # ======================================================================= #
    model = DenseResNet(replace(params, name=f"DenseResNet{NAME}"))
    results.append((model, *train_model(model, x_t, y_t)))
    
    # ======================================================================= #
    # ConvResNet                                                              #
    # ======================================================================= #
    model = ConvResNet(replace(params, name=f"ConvResNet{NAME}"), kernel_size=K_SIZE, padding=PADDING, stride=STRIDE, connect=CONNECT)
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
    model = CustomNet(replace(params, name=f"CustomNet{NAME}"), nodes)
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