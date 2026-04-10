import torch.nn as nn
import torch

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