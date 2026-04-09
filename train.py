import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

class DenseNet(nn.Module):
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

def update(frame_idx):
    epoch, mse_val, ce_val, y_pred = snapshots[frame_idx]

    # Fit panel
    line_pred.set_data(x_np, y_pred)
    title_text.set_text(f"Epoch {epoch:>4d}")

    # MSE loss panel
    mask = epoch_vals <= epoch
    line_mse.set_data(epoch_vals[mask], np.array(mse_losses)[mask])
    dot_mse.set_data([epoch], [mse_val])
 
    # Cross-entropy loss panel
    line_ce.set_data(epoch_vals[mask], np.array(ce_losses)[mask])
    dot_ce.set_data([epoch], [ce_val])
 
    return line_pred, title_text, line_mse, dot_mse, line_ce, dot_ce

if __name__ == "__main__":
    # Configs
    H_N    = 3              # number of hidden layers
    N_N    = 64             # nodes per hidden layer
    FUNC   = nn.Tanh()      # activation: nn.Tanh() | nn.ReLU() | nn.SiLU() | etc.
    EPOCHS = 2000
    LR     = 1e-3
    LOG_EVERY = 50          # snapshot + print every N epochs
    
    # Synthetic data
    x_np = np.linspace(-3, 3, 200).astype(np.float32)
    y_np = np.sin(x_np) + 0.2 * np.random.randn(*x_np.shape).astype(np.float32)

    x = torch.tensor(x_np).unsqueeze(1)   # (200, 1)
    y = torch.tensor(y_np).unsqueeze(1)   # (200, 1)
    
    # Build Model
    model     = DenseNet(h_n=H_N, n_n=N_N, func=FUNC)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_fn    = nn.MSELoss()
    
    print(model)
    print(f"\nTraining  |  layers={H_N}  nodes={N_N}  act={FUNC.__class__.__name__}  epochs={EPOCHS}\n")

    # Training loop (collect snapshots)
    snapshots  = []   # (epoch, mse, ce, y_pred)
    mse_losses = []
    ce_losses  = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        mse  = mse_fn(pred, y)
        mse.backward()
        optimizer.step()

        mse_val = mse.item()
        mse_losses.append(mse_val)
        
        # Cross-entropy proxy: treat normalised predictions as a prob distribution.
        with torch.no_grad():
            p     = torch.softmax(pred.squeeze(), dim=0)
            q     = torch.softmax(y.squeeze(),    dim=0)
            ce_val = -torch.sum(q * torch.log(p + 1e-9)).item()
        ce_losses.append(ce_val)
 
        if epoch % LOG_EVERY == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x).squeeze().numpy()
            snapshots.append((epoch, mse_val, ce_val, y_pred.copy()))
            print(f"Epoch {epoch:>4d}  |  MSE: {mse_val:.6f}  |  CE: {ce_val:.4f}")

    epoch_vals = np.arange(1, EPOCHS + 1)
 
    # ── Figure layout (1 row, 3 panels) ───────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
 
    ax_fit  = fig.add_subplot(gs[0])
    ax_mse  = fig.add_subplot(gs[1])
    ax_ce   = fig.add_subplot(gs[2])
 
    # Fit panel — static scatter
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa", label="Data", zorder=1)
    ax_fit.set_xlim(-3.2, 3.2)
    ax_fit.set_ylim(-1.8, 1.8)
    ax_fit.set_xlabel("x")
    ax_fit.set_ylabel("y")
 
    # MSE panel
    ax_mse.set_xlim(1, EPOCHS)
    ax_mse.set_ylim(max(min(mse_losses) * 0.5, 1e-9), max(mse_losses) * 1.1)
    ax_mse.set_yscale("log")
    ax_mse.set_xlabel("Epoch")
    ax_mse.set_ylabel("MSE Loss")
    ax_mse.set_title("MSE Loss")
 
    # Cross-entropy panel
    ax_ce.set_xlim(1, EPOCHS)
    ax_ce.set_ylim(max(min(ce_losses) * 0.5, 1e-9), max(ce_losses) * 1.1)
    ax_ce.set_yscale("log")
    ax_ce.set_xlabel("Epoch")
    ax_ce.set_ylabel("Cross-Entropy Loss")
    ax_ce.set_title("Cross-Entropy Loss")
 
    # Dynamic artists
    (line_pred,) = ax_fit.plot([], [], lw=2, color="#e05c2e", zorder=2, label="Prediction")
    title_text   = ax_fit.set_title("")
    ax_fit.legend(loc="upper right", fontsize=8)
 
    (line_mse,) = ax_mse.plot([], [], lw=1.5, color="#2e7de0")
    dot_mse,    = ax_mse.plot([], [], "o", color="#e05c2e", ms=6, zorder=3)
 
    (line_ce,)  = ax_ce.plot([], [], lw=1.5, color="#7c3aed")
    dot_ce,     = ax_ce.plot([], [], "o", color="#e05c2e", ms=6, zorder=3)
 
    fig.suptitle(
        f"Dense NN Curve Fitting  —  {H_N} hidden layers × {N_N} nodes  [{FUNC.__class__.__name__}]",
        fontsize=12,
    )
    fig.subplots_adjust(top=0.88, wspace=0.4)
 
    ani = FuncAnimation(
        fig, update,
        frames=len(snapshots),
        interval=80,
        blit=True,
        repeat=False,
    )
 
    plt.show()
 
    # Uncomment to save:
    ani.save("training.gif", writer="pillow", fps=15)
