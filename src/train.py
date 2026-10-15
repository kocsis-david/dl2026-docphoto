"""Train the MLP on raw pixels and log the run to Weights & Biases.

    python -m src.train --config configs/lab3_mlp.yaml

Given: pixel standardisation, the epoch loop with its logging, and the
checkpoint. Yours, each a few lines, each with its contract in the docstring:

    make_optimizer   TASK 3
    train_epoch      TASK 4   the minibatch loop
    predict          TASK 5   model outputs as numpy
    validate         TASK 6   RMSE on the scoreable validation rows

Read the given parts once. There is no framework underneath: this file is the
whole of supervised training.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, tracking
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.evaluate import rmse  # noqa: F401  (TASK 6 needs it)
from src.models.mlp import MLP
from src.seed import set_seed


class Standardiser:
    """Zero mean and unit variance over all pixels, fitted on the training rows only.

    One mean and one scale for the whole image, not one per pixel: with a
    thousand training rows a per-pixel estimate is mostly noise. Fitting on
    all rows would leak the validation distribution into training; a small
    leak, but the habit matters. The two numbers go into the checkpoint so
    that prediction uses exactly the same transform.
    """

    def fit(self, X: np.ndarray) -> Standardiser:
        self.mean = float(X.mean())
        self.scale = float(X.std()) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


# --------------------------------------------------------------- your part

def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """TASK 3. The optimiser, using cfg['train']['lr'] and cfg['train']['weight_decay']."""
    return torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """TASK 4. One pass over the training rows in shuffled minibatches.

    Draw the order with `torch.randperm(len(X), generator=generator)` so the
    run is reproducible, step the optimiser once per minibatch, and return the
    mean training loss over the epoch, weighted by minibatch size.
    """
    model.train()
    order = torch.randperm(len(X), generator=generator)
    total = torch.zeros((), device=X.device)
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size].to(X.device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.detach() * len(idx)
    return float(total) / len(X)


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """TASK 5. The model's outputs for every row of X, as a numpy array of shape (len(X),).

    Evaluation mode, no gradients, and it must give the same answer for a row
    whether that row is predicted alone or inside a batch.
    """
    model.eval()
    with torch.no_grad():
        out = [model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.cat(out).cpu().numpy()


def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    """TASK 6. RMSE on the validation rows where `ok` is True (the scoreable ones)."""
    pred = predict(model, X_val)
    return rmse(y_val[ok], pred[ok])


# ------------------------------------------------------------------ given

def train_mlp(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              device: str | None = None, tags: list[str] | None = None, log=print) -> dict:
    """Train on the pixel rows train_idx, validate on val_idx after every epoch, log everything.

    `X` is the pixel table, one row per row of `df`. Returns the history, the
    validation predictions after the first and the last epoch, the trained
    model, the checkpoint path and the W&B run page.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]
    split = cfg["split"]

    fit_pos, inner_pos = build_splits(df.iloc[train_idx], split["group_column"], split["val_fraction"], split["seed"])
    fit_idx, inner_idx = train_idx[fit_pos], train_idx[inner_pos]

    scaler = Standardiser().fit(X[fit_idx])
    X_fit = to_tensor(scaler.transform(X[fit_idx]), device)
    X_in = to_tensor(scaler.transform(X[inner_idx]), device)
    X_va = to_tensor(scaler.transform(X[val_idx]), device)
    y_fit = to_tensor(df[config.TARGET].to_numpy()[fit_idx], device)
    y_in = df[config.TARGET].to_numpy()[inner_idx]
    y_va = df[config.TARGET].to_numpy()[val_idx]
    ok_in = df["eval_ok"].to_numpy()[inner_idx] == 1
    ok = df["eval_ok"].to_numpy()[val_idx] == 1

    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=cfg["model"]["dropout"],
                prior=float(y_fit.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()           # the metric is RMSE; same minimiser
    generator = torch.Generator().manual_seed(cfg["seed"])

    tracking.init_run(cfg, tags=tags)
    history, snapshots = [], {}
    best_inner, best_epoch, best_state = float("inf"), None, None
    log(f"{cfg['name']}: {len(fit_idx)} fit / {len(inner_idx)} inner / {len(val_idx)} val rows, {X.shape[1]} inputs, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters, {device}")
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, loss_fn, X_fit, y_fit, tr["batch_size"], generator)
        inner = validate(model, X_in, y_in, ok_in)
        val = validate(model, X_va, y_va, ok)
        train_eval = rmse(y_fit.cpu().numpy(), predict(model, X_fit))   # dropout off, as in validation
        row = {"epoch": epoch, "train_loss": loss, "train_rmse": train_eval, "inner_rmse": inner,
               "val_rmse": val, "seconds": time.time() - t0}
        history.append(row)
        if inner < best_inner:
            best_inner, best_epoch = inner, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tracking.log_metrics({"train_loss": loss, "train_rmse": train_eval, "inner_rmse": inner,
                              "val_rmse": val}, step=epoch)
        if epoch in (1, tr["epochs"]) or epoch % tr.get("log_every", 10) == 0:
            log(f"epoch {epoch:3d}  train loss {loss:.4f}  train RMSE {train_eval:.4f}  "
                f"inner RMSE {inner:.4f}  val RMSE {val:.4f}")
        if epoch in (1, tr["epochs"]):
            snapshots[epoch] = predict(model, X_va)

    model.load_state_dict(best_state)
    history = pd.DataFrame(history)
    selected_val = float(history.loc[best_epoch - 1, "val_rmse"])
    log(f"early stopping: epoch {best_epoch} chosen on the inner split (RMSE {best_inner:.4f}); "
        f"validation RMSE at that epoch {selected_val:.4f}")

    ckpt = config.ROOT / "models" / f"{cfg['name']}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "in_dim": X.shape[1],
                "scaler_mean": scaler.mean, "scaler_scale": scaler.scale, "best_epoch": best_epoch}, ckpt)
    tracking.log_checkpoint(ckpt, cfg["name"], {"val_rmse": selected_val, "best_epoch": best_epoch})
    tracking.log_metrics({"final_val_rmse": selected_val, "best_epoch": best_epoch, "inner_rmse": best_inner})
    url = tracking.run_url()
    tracking.finish()
    return {"history": history, "val_pred": predict(model, X_va), "snapshots": snapshots,
            "model": model, "checkpoint": ckpt, "url": url, "val_index": val_idx, "ok": ok,
            "best_epoch": best_epoch}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLP from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"], cfg["split"]["seed"])
    result = train_mlp(cfg, X, df, train_idx, val_idx)
    print(f"final val RMSE {result['history']['val_rmse'].iloc[-1]:.4f}  checkpoint {result['checkpoint']}  "
          f"run {result['url']}")


if __name__ == "__main__":
    main()
