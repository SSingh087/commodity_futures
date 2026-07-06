"""
merge_chunks.py

Concatenates all options_on_futures_data_chunk_*.pt files produced by
the parallel generation jobs into one options_on_futures_data.pt file.

Reads data paths from the same config the workers used, so nothing is
hard-coded.  Optionally checks that exactly N_CHUNKS files arrived before
merging — useful as a DAG sanity check.

Usage:
    python merge_chunks.py --config ./config/SchwartzSmithFWD.yaml --n_chunks 100
"""

import argparse
import glob
import os
import sys
import yaml
import torch
import numpy as np


def merge(data_dir: str, out_path: str, n_chunks_expected: int | None) -> None:
    pattern     = os.path.join(data_dir, "options_on_futures_data_chunk_*.pt")
    chunk_paths = sorted(glob.glob(pattern))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found matching: {pattern}")

    # Guard: fail loudly if DAG finished early (e.g. some workers were evicted)
    if n_chunks_expected is not None and len(chunk_paths) != n_chunks_expected:
        raise RuntimeError(
            f"Expected {n_chunks_expected} chunks, found {len(chunk_paths)}. "
            "Some worker jobs may have failed — check ERROR_FILES/."
        )

    print(f"Found {len(chunk_paths)} chunks — merging...")

    X_all, V_all, V_std_all = [], [], []
    col_names = None

    for path in chunk_paths:
        d = torch.load(path, weights_only=True)
        X_all.append(d["X"])
        V_all.append(d["V"])
        V_std_all.append(d["V_std"])
        if col_names is None:
            col_names = d["col_names"]

    X     = torch.cat(X_all,     dim=0)
    V     = torch.cat(V_all,     dim=0)
    V_std = torch.cat(V_std_all, dim=0)

    print(f"Total samples : {len(X):,}")
    print(f"Price range   : [{V.min():.4f}, {V.max():.4f}]")
    print(f"Mean price    : {V.mean():.4f}")

    # Recompute normalisation over the full merged dataset
    X_np       = X.numpy()
    V_np       = V.numpy()
    X_mean     = X_np.mean(axis=0).astype(np.float32)
    X_std      = (X_np.std(axis=0) + 1e-8).astype(np.float32)
    V_mean     = float(V_np.mean())
    V_std_norm = float(V_np.std()) + 1e-8

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        "X"          : X,
        "V"          : V,
        "V_std"      : V_std,
        "col_names"  : col_names,
        "X_mean"     : torch.tensor(X_mean),
        "X_std"      : torch.tensor(X_std),
        "V_mean"     : V_mean,
        "V_std_norm" : V_std_norm,
    }, out_path)
    print(f"Merged dataset saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="./config/SchwartzSmithFWD.yaml")
    parser.add_argument("--n_chunks", type=int, default=None,
                        help="Expected number of chunk files (optional sanity check)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["save_path"]
    out_path = os.path.join(data_dir, "options_on_futures_data.pt")

    merge(data_dir, out_path, args.n_chunks)