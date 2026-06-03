#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _is_scalar(x: Any) -> bool:
    # np scalar types + python scalars
    return isinstance(x, (int, float, complex, np.generic))


def _first_item(x: Any) -> Any:
    """Return the first item of an iterable container, or raise TypeError."""
    if isinstance(x, dict):
        # deterministic first key (sorted) to avoid randomness
        k = sorted(x.keys())[0]
        return x[k]
    if isinstance(x, (list, tuple)):
        return x[0]
    if isinstance(x, np.ndarray):
        return x.flat[0] if x.size else np.nan
    raise TypeError(f"Not indexable: {type(x)}")


def to_scalar(x: Any, max_depth: int = 10) -> float:
    """
    Your metrics are stored as nested lists/tuples/arrays (pol->channel->maybe [value]).
    This attempts to peel layers until a scalar is found.
    """
    cur = x
    for _ in range(max_depth):
        if cur is None:
            return np.nan
        if _is_scalar(cur):
            # convert complex to magnitude if it ever happens
            if isinstance(cur, complex):
                return float(np.abs(cur))
            return float(cur)
        # numpy 0-d arrays
        if isinstance(cur, np.ndarray) and cur.shape == ():
            v = cur.item()
            return float(np.abs(v)) if isinstance(v, complex) else float(v)
        try:
            cur = _first_item(cur)
        except Exception:
            break

    # last attempt: try numeric cast
    try:
        return float(cur)
    except Exception:
        return np.nan


def load_pickles(data_dir: Path, job_name: str, n_channels: Optional[int] = None) -> pd.DataFrame:
    patt = f"data_collected_{job_name}_nch_*_pavedbm_*.pkl"
    files = sorted(data_dir.glob(patt))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {patt}")

    dfs: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_pickle(fp)
        if n_channels is not None and "n_channels" in df.columns:
            df = df[df["n_channels"] == n_channels]
        df["__source_file"] = fp.name
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def add_scalar_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Convert nested structures into simple float columns
    mapping = {
        "ber": "ber_s",
        "q": "q_s",
        "evm": "evm_s",
        "mi": "mi_s",
        "ber_w_noise": "ber_w_s",
        "q_w_noise": "q_w_s",
        "evm_w_noise": "evm_w_s",
        "mi_w_noise": "mi_w_s",
    }
    for src, dst in mapping.items():
        if src in df.columns:
            df[dst] = df[src].apply(to_scalar)
        else:
            df[dst] = np.nan
    return df


def print_overview(df: pd.DataFrame) -> None:
    print("\n=== Loaded dataset ===")
    print(f"Rows: {len(df)}")
    show_cols = [c for c in ["job_name", "n_channels", "n_polarisations", "n_symbols", "noise_figure_db",
                            "n_span", "z_km", "p_ave_dbm", "__source_file"] if c in df.columns]
    if show_cols:
        print(df[show_cols].head(10).to_string(index=False))

    scalar_cols = [c for c in ["ber_s", "q_s", "evm_s", "mi_s", "ber_w_s", "q_w_s", "evm_w_s", "mi_w_s"] if c in df.columns]
    if scalar_cols:
        print("\n=== Scalar metric describe() ===")
        print(df[scalar_cols].describe().to_string())

    if "p_ave_dbm" in df.columns:
        print("\n=== Counts per p_ave_dbm ===")
        print(df["p_ave_dbm"].value_counts().sort_index().to_string())


def plot_metric_vs_power(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """
    Plot mean ± std per p_ave_dbm for a scalar metric column.
    """
    if "p_ave_dbm" not in df.columns:
        print("No p_ave_dbm column; skipping plots.")
        return
    if metric not in df.columns or df[metric].isna().all():
        print(f"Metric {metric} missing/empty; skipping.")
        return

    g = df.groupby("p_ave_dbm")[metric]
    stats = g.agg(["mean", "std", "count"]).reset_index().sort_values("p_ave_dbm")

    x = stats["p_ave_dbm"].to_numpy(dtype=float)
    y = stats["mean"].to_numpy(dtype=float)
    e = stats["std"].to_numpy(dtype=float)

    plt.figure()
    plt.errorbar(x, y, yerr=e, fmt="o-", capsize=3)
    plt.xlabel("Launch power, p_ave_dbm (dBm)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs launch power (mean ± std)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out_path = out_dir / f"{metric}_vs_pave_dbm.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/home/esf0/data/errorstat_channel_noise/"))
    ap.add_argument("--job-name", type=str, default="test")
    ap.add_argument("--n-channels", type=int, default=None, help="Optional filter for n_channels")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to save figures (default: data-dir/plots_<job>)")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    if args.out_dir is None:
        out_dir = data_dir / f"plots_{args.job_name}"
    else:
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pickles(data_dir=data_dir, job_name=args.job_name, n_channels=args.n_channels)
    df = add_scalar_metrics(df)

    # Optional: keep a cleaned copy for further work
    cleaned_path = out_dir / f"combined_{args.job_name}.parquet"
    try:
        df.to_parquet(cleaned_path, index=False)
        print(f"Saved combined parquet: {cleaned_path}")
    except Exception as e:
        print(f"(Could not save parquet: {e})")

    print_overview(df)

    # Plots
    for m in ["ber_s", "q_s", "evm_s", "mi_s", "ber_w_s", "q_w_s", "evm_w_s", "mi_w_s"]:
        plot_metric_vs_power(df, m, out_dir)

    # Also print a compact table per power
    if "p_ave_dbm" in df.columns:
        cols = [c for c in ["ber_s", "q_s", "evm_s", "mi_s", "ber_w_s", "q_w_s", "evm_w_s", "mi_w_s"] if c in df.columns]
        if cols:
            summary = df.groupby("p_ave_dbm")[cols].agg(["mean", "std", "count"]).round(6)
            print("\n=== Summary by p_ave_dbm (mean/std/count) ===")
            print(summary.to_string())

            csv_path = out_dir / f"summary_{args.job_name}.csv"
            summary.to_csv(csv_path)
            print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
