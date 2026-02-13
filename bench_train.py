
import os
import json
import time
import random
from typing import List, Union

import numpy as np
import pandas as pd
import torch

from bench_utils import DeclareArg
from ddpm import TransEncoder, GaussianDiffusion1D

from long_discriminative_score import long_discriminative_score_metrics
from long_predictive_score import long_predictive_score_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_etth1(csv_path: str, columns: List[str]) -> np.ndarray:
    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. Available: {list(df.columns)}")

    x = df[columns].astype(np.float32).to_numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x  # (T, D)


def ett_split_indices(T: int, seq_len: int):
    train_end = 12 * 30 * 24  # 8640
    val_end = train_end + 4 * 30 * 24  # 11520
    test_end = val_end + 4 * 30 * 24  # 14400

    if T >= test_end:
        tr0, tr1 = 0, train_end
        va0, va1 = max(0, train_end - seq_len), val_end
        te0, te1 = max(0, val_end - seq_len), test_end
        return (tr0, tr1), (va0, va1), (te0, te1)

    tr1 = int(0.60 * T)
    va1 = int(0.80 * T)
    tr0 = 0
    va0 = max(0, tr1 - seq_len)
    te0 = max(0, va1 - seq_len)
    te1 = T
    return (tr0, tr1), (va0, va1), (te0, te1)


def make_windows(x_2d: np.ndarray, seq_len: int) -> np.ndarray:
    T, D = x_2d.shape
    if T < seq_len:
        raise ValueError(f"Not enough length T={T} for seq_len={seq_len}")
    N = T - seq_len + 1
    out = np.empty((N, seq_len, D), dtype=np.float32)
    for i in range(N):
        out[i] = x_2d[i : i + seq_len]
    return out


def fit_minmax(train_2d: np.ndarray):
    mn = train_2d.min(axis=0)
    mx = train_2d.max(axis=0)
    denom = (mx - mn).astype(np.float32)
    denom[denom == 0.0] = 1.0
    return mn.astype(np.float32), denom


def apply_minmax(x_2d: np.ndarray, mn: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return ((x_2d - mn) / denom).astype(np.float32)


def parse_num_samples(num_samples_arg: str) -> Union[str, int]:
    s = str(num_samples_arg).strip().lower()
    if s in ("same", "train", "same_train"):
        return "same"
    try:
        v = int(s)
        if v <= 0:
            raise ValueError
        return v
    except Exception:
        raise ValueError(f'num_samples must be "same" or a positive integer. Got: {num_samples_arg}')


def resolve_device(device_arg: str) -> torch.device:
    d = str(device_arg).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    seed = DeclareArg("seed", int, 42, "random seed")

    data_csv = DeclareArg(
        "data_csv",
        str,
        r"D:\TimeSeries-Generative-Modeling-RnD\data\ETT-small\ETTh1.csv",
        "path to ETTh1.csv",
    )

    seq_len = DeclareArg("seq_len", int, 96, "window length")
    columns_str = DeclareArg(
        "columns",
        str,
        "HUFL,HULL,MUFL,MULL,LUFL,LULL,OT",
        "comma-separated columns to use",
    )
    columns = [c.strip() for c in columns_str.split(",") if c.strip()]

    latent_dim = DeclareArg("latent_dim", int, 128, "transformer hidden size")
    n_heads = DeclareArg("n_heads", int, 8, "attention heads")
    num_layers = DeclareArg("num_layers", int, 6, "transformer layers")
    ff_size = DeclareArg("ff_size", int, 1024, "feedforward size")
    dropout = DeclareArg("dropout", float, 0.1, "dropout")
    activation = DeclareArg("activation", str, "gelu", "activation (relu/gelu)")

    timesteps = DeclareArg("timesteps", int, 1000, "diffusion training timesteps")
    sampling_timesteps = DeclareArg("sampling_timesteps", int, 250, "sampling steps")
    beta_schedule = DeclareArg("beta_schedule", str, "cosine", "beta schedule")
    objective = DeclareArg("objective", str, "pred_noise", "objective")
    loss_type = DeclareArg("loss_type", str, "l2", "loss type")

    epochs = DeclareArg("epochs", int, 1, "training epochs")
    batch_size = DeclareArg("batch_size", int, 32, "batch size")
    learning_rate = DeclareArg("learning_rate", float, 1e-4, "learning rate")
    max_steps = DeclareArg("max_steps", int, 0, "0 = run full epoch, else stop after N steps")

    num_samples_arg = DeclareArg("num_samples", str, "512", '"same" or integer number of samples')
    max_samples = DeclareArg("max_samples", int, 2000, "cap maximum generated samples")

    metrics_max_samples = DeclareArg("metrics_max_samples", int, 2000, "cap samples used for metrics")

    save_dir = DeclareArg("save_dir", str, "./results/bench/transfusion", "output folder")
    device_arg = DeclareArg("device", str, "auto", "auto/cpu/cuda")

    set_seed(seed)
    device = resolve_device(device_arg)

    setting = f"TransFusion_ETTh1_sl{seq_len}_ld{latent_dim}_h{n_heads}_nl{num_layers}_seed{seed}"
    run_dir = os.path.join(save_dir, setting)
    os.makedirs(run_dir, exist_ok=True)

    x = load_etth1(data_csv, columns)  # (T, D)
    T, D = x.shape
    (tr0, tr1), (va0, va1), (te0, te1) = ett_split_indices(T, seq_len)

    train_2d = x[tr0:tr1]
    val_2d = x[va0:va1]
    test_2d = x[te0:te1]

    mn, denom = fit_minmax(train_2d)
    train_s = apply_minmax(train_2d, mn, denom)
    val_s = apply_minmax(val_2d, mn, denom)
    test_s = apply_minmax(test_2d, mn, denom)

    train_w = make_windows(train_s, seq_len)  # (N, L, D)
    val_w = make_windows(val_s, seq_len)
    test_w = make_windows(test_s, seq_len)

    train_t = torch.from_numpy(train_w).permute(0, 2, 1).contiguous()  # (N, D, L)
    train_loader = torch.utils.data.DataLoader(train_t, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TransEncoder(
        features=D,
        latent_dim=latent_dim,
        num_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        ff_size=ff_size,
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=seq_len,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type=loss_type,
        beta_schedule=beta_schedule,
    ).to(device)

    optim = torch.optim.Adam(diffusion.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    t0 = time.time()
    diffusion.train()
    steps_done = 0

    for _ in range(int(epochs)):
        for batch in train_loader:
            batch = batch.to(device)

            optim.zero_grad(set_to_none=True)
            loss = diffusion(batch)
            loss.backward()
            optim.step()

            steps_done += 1

            if steps_done == 1 or steps_done % 50 == 0:
                print(f"step: {steps_done} loss: {loss.item():.6f}")

            if int(max_steps) > 0 and steps_done >= int(max_steps):
                break

        if int(max_steps) > 0 and steps_done >= int(max_steps):
            break

    train_time_s = time.time() - t0

    diffusion.eval()

    num_samples_parsed = parse_num_samples(num_samples_arg)
    if num_samples_parsed == "same":
        sample_n = int(min(train_w.shape[0], max_samples))
    else:
        sample_n = int(min(int(num_samples_parsed), max_samples))

    t1 = time.time()
    with torch.no_grad():
        synth_c_l = diffusion.sample(batch_size=sample_n)  # (B, C, L)
    sample_time_s = time.time() - t1

    synth_l_d = synth_c_l.permute(0, 2, 1).cpu().numpy().astype(np.float32)  # (B, L, D)

    m = int(metrics_max_samples)
    real_for_metrics = train_w[:m]
    synth_for_metrics = synth_l_d[:m]

    if synth_for_metrics.shape[0] < real_for_metrics.shape[0]:
        real_for_metrics = real_for_metrics[: synth_for_metrics.shape[0]]
    elif synth_for_metrics.shape[0] > real_for_metrics.shape[0]:
        synth_for_metrics = synth_for_metrics[: real_for_metrics.shape[0]]

    real_t = torch.from_numpy(real_for_metrics).float()
    synth_t = torch.from_numpy(synth_for_metrics).float()

    disc_train = float(long_discriminative_score_metrics(real_t, synth_t))
    pred_mae = float(long_predictive_score_metrics(real_t, synth_t))

    np.save(os.path.join(run_dir, "synthetic_samples.npy"), synth_l_d[: min(2000, synth_l_d.shape[0])])
    np.save(os.path.join(run_dir, "real_train_head.npy"), train_w[:50])
    np.save(os.path.join(run_dir, "real_test_head.npy"), test_w[:50])
    np.save(os.path.join(run_dir, "min_train.npy"), mn)
    np.save(os.path.join(run_dir, "denom_train.npy"), denom)

    run_args = {
        "seed": int(seed),
        "data_csv": os.path.abspath(data_csv),
        "columns": columns,
        "seq_len": int(seq_len),
        "latent_dim": int(latent_dim),
        "n_heads": int(n_heads),
        "num_layers": int(num_layers),
        "ff_size": int(ff_size),
        "dropout": float(dropout),
        "activation": str(activation),
        "timesteps": int(timesteps),
        "sampling_timesteps": int(sampling_timesteps),
        "beta_schedule": str(beta_schedule),
        "objective": str(objective),
        "loss_type": str(loss_type),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_steps": int(max_steps),
        "num_samples": str(num_samples_arg),
        "max_samples": int(max_samples),
        "metrics_max_samples": int(metrics_max_samples),
        "save_dir": str(save_dir),
        "device": str(device),
    }

    with open(os.path.join(run_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2)

    metrics = {
        "setting": setting,
        "model": "TransFusion",
        "dataset": "ETTh1",
        "shapes": {
            "raw_T": int(T),
            "raw_D": int(D),
            "train_w": list(train_w.shape),
            "val_w": list(val_w.shape),
            "test_w": list(test_w.shape),
            "synthetic": list(synth_l_d.shape),
        },
        "metrics": {
            "discriminative_train": disc_train,
            "predictive_mae": pred_mae,
        },
        "timing_seconds": {
            "train": float(train_time_s),
            "sample": float(sample_time_s),
        },
        "notes": {
            "scaling": "min-max scaling fit on train split",
            "metrics_space": "metrics computed on scaled [0,1] windows",
        },
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"done: {setting} train_s={train_time_s:.2f} sample_s={sample_time_s:.2f} "
        f"disc_train={disc_train:.6f} pred_mae={pred_mae:.6f} out={os.path.abspath(run_dir)}"
    )


if __name__ == "__main__":
    main()
