


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from ddpm import TransEncoder, GaussianDiffusion1D



CSV_PATH = r"D:\TimeSeries-Generative-Modeling-RnD\data\ETT-small\ETTh1.csv"
COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

SEQ_LEN = 24
BATCH_SIZE = 64
MAX_STEPS = 50

TIMESTEPS = 50
BETA_SCHEDULE = "cosine"       # "cosine" | "linear" | "quadratic" | "sigmoid"
OBJECTIVE = "pred_v"           # "pred_x0" | "pred_v" | "pred_noise"

HIDDEN_DIM = 128
N_HEAD = 4
NUM_LAYERS = 4

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

SEED = 123
USE_CPU = True                
SAVE_DIR = "results_quicktest_transfusion"
NUM_SAMPLES = 16


def set_seed(seed: int = 123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(arr_2d: np.ndarray, seq_len: int) -> np.ndarray:
    """
    arr_2d: (T, D)
    return: (N, L, D) where N = T - L + 1
    """
    T, D = arr_2d.shape
    if T < seq_len:
        raise ValueError(f"Not enough rows T={T} for seq_len={seq_len}")
    N = T - seq_len + 1
    out = np.empty((N, seq_len, D), dtype=np.float32)
    for i in range(N):
        out[i] = arr_2d[i:i + seq_len]
    return out


def minmax_fit(train_windows: np.ndarray, eps: float = 1e-8):
    """
    train_windows: (N, L, D)
    Returns per-feature min and scale over ALL train points.
    """
    flat = train_windows.reshape(-1, train_windows.shape[-1])  # (N*L, D)
    fmin = flat.min(axis=0)
    fmax = flat.max(axis=0)
    scale = np.maximum(fmax - fmin, eps)
    return fmin, scale


def minmax_transform(windows: np.ndarray, fmin: np.ndarray, scale: np.ndarray):
    """
    transform to [0,1]
    """
    return (windows - fmin[None, None, :]) / scale[None, None, :]


def add_time_channel(windows: np.ndarray) -> np.ndarray:
    """
    windows: (N, L, D)
    returns: (N, L, D+1) with time in last channel in [0,1]
    """
    N, L, D = windows.shape
    t = np.linspace(0.0, 1.0, L, dtype=np.float32)[None, :, None]  # (1,L,1)
    t = np.repeat(t, N, axis=0)  # (N,L,1)
    return np.concatenate([windows, t], axis=2)  # (N,L,D+1)


@torch.no_grad()
def sample_and_save(diffusion, out_dir, batch_size, num_samples):
    diffusion.eval()

    all_samples = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        x = diffusion.sample(batch_size=b)  # (b, c, n) in [-1,1]
        x = x.detach().cpu().numpy()
        all_samples.append(x)
        remaining -= b

    samples = np.concatenate(all_samples, axis=0)  # (num_samples, c, n)

    # save to [0,1] for inspection
    samples_01 = (samples + 1.0) / 2.0
    samples_01 = np.clip(samples_01, 0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "samples_raw_neg1_to_1.npy"), samples)
    np.save(os.path.join(out_dir, "samples_0_to_1.npy"), samples_01)

    print(f"[OK] Saved samples:")
    print(f"  {os.path.join(out_dir, 'samples_raw_neg1_to_1.npy')}  shape={samples.shape}")
    print(f"  {os.path.join(out_dir, 'samples_0_to_1.npy')}          shape={samples_01.shape}")


def main():
    set_seed(SEED)

    device = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print("=== TransFusion Quick Test on ETTh1 ===")
    print(f"CSV: {CSV_PATH}")
    print(f"SEQ_LEN: {SEQ_LEN}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[COLS].dropna()

    raw = df.values.astype(np.float32)  # (T, D)
    T, D = raw.shape

    print(f"Columns: {COLS}")
    print(f"Raw series shape: T={T}, D={D}")

    # contiguous split
    n_train = int(T * TRAIN_RATIO)
    n_val = int(T * VAL_RATIO)

    train_range = (0, n_train)
    val_range = (n_train - SEQ_LEN, n_train + n_val)
    test_range = (n_train + n_val - SEQ_LEN, T)

    print("Split indices:")
    print(f"  Train: {train_range[0]} -> {train_range[1]}")
    print(f"  Val:   {val_range[0]} -> {val_range[1]}")
    print(f"  Test:  {test_range[0]} -> {test_range[1]}")

    train_series = raw[train_range[0]:train_range[1]]
    test_series = raw[test_range[0]:test_range[1]]

    train_windows = make_windows(train_series, SEQ_LEN)
    test_windows = make_windows(test_series, SEQ_LEN)

    print("\nWindowed shapes (N, L, D):")
    print(f"  train_windows: {train_windows.shape}")
    print(f"  test_windows:  {test_windows.shape}")

    # scale using train only
    fmin, scale = minmax_fit(train_windows)
    train_windows_01 = minmax_transform(train_windows, fmin, scale)
    test_windows_01 = minmax_transform(test_windows, fmin, scale)

    # add time channel
    train_windows_t = add_time_channel(train_windows_01)
    test_windows_t = add_time_channel(test_windows_01)

    print("\nAfter adding time channel (N, L, D+1):")
    print(f"  train_windows_t: {train_windows_t.shape}")
    print(f"  test_windows_t:  {test_windows_t.shape}")

    # TransFusion expects (N, C, L)
    train_tensor = torch.from_numpy(train_windows_t).permute(0, 2, 1).contiguous()
    test_tensor = torch.from_numpy(test_windows_t).permute(0, 2, 1).contiguous()

    N, C, L = train_tensor.shape
    print("\nModel input shape (N, C, L):")
    print(f"  train_tensor: {train_tensor.shape}")
    print(f"  test_tensor:  {test_tensor.shape}")

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    model = TransEncoder(
        features=C,
        latent_dim=HIDDEN_DIM,
        num_heads=N_HEAD,
        num_layers=NUM_LAYERS
    ).to(device)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length=L,
        timesteps=TIMESTEPS,
        objective=OBJECTIVE,
        beta_schedule=BETA_SCHEDULE,
    ).to(device)

    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    print("\n=== Starting Quick Training ===")
    diffusion.train()

    step = 0
    for (batch,) in loader:
        batch = batch.to(device)          # (B, C, L) in [0,1]
        loss = diffusion(batch)           # diffusion handles internal transforms

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        step += 1
        if step == 1 or step % 5 == 0:
            print(f"step {step}/{MAX_STEPS}  loss={loss.item():.6f}")

        if step >= MAX_STEPS:
            break

    print("=== Training Done ===")

    sample_and_save(
        diffusion=diffusion,
        out_dir=SAVE_DIR,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES
    )

    print("\n[OK] Quicktest finished successfully.")


if __name__ == "__main__":
    main()