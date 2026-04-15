import torch.nn as nn
import torch
import numpy as np

def _snapshot_weights(model: nn.Module) -> list[np.ndarray]:
    """
    Return a list of weight matrices for every Linear / Conv1d layer in the
    model, in the same order that animation.extract_layer_weights() produces.
    Conv1d weights are averaged over the kernel dimension so every matrix has
    shape (out, in).
    """
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().copy())
        elif isinstance(module, nn.Conv1d):
            w = module.weight.detach().cpu().numpy()
            weights.append(w.mean(axis=2).copy())
    return weights

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
        snapshots  : list of (epoch, mse, bce, y_pred_np, weights)
                     where weights is a list of np.ndarray, one per
                     Linear / Conv1d layer (kernel dim averaged for Conv1d).
        mse_losses : MSE at every epoch
        ce_losses  : binary cross-entropy proxy at every epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn    = nn.MSELoss()
    ce_fn     = nn.BCEWithLogitsLoss()

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
        ce   = ce_fn(pred, y)
        mse.backward()
        optimizer.step()

        mse_val = mse.item()
        mse_losses.append(mse_val)
        
        ce_val = ce.item()
        ce_losses.append(ce_val)

        if epoch % log_every == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x).squeeze().cpu().numpy()
            snapshots.append((epoch, mse_val, ce_val, y_pred.copy(), _snapshot_weights(model)))
            print(f"Epoch {epoch:>4d}  |  MSE: {mse_val:.6f}  |  CE: {ce_val:.4f}")
            
    return snapshots, mse_losses, ce_losses