"""Student Performance real-data experiment.

Uses student-por.csv (649 observations), with G1, G2, and G3 as the
three-dimensional response. The grade columns are excluded from X to avoid
target leakage; categorical predictors are one-hot encoded.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import run_energy_efficiency_realdata as base


def load_student(path):
    df = pd.read_csv(path, sep=";")
    response_cols = ["G1", "G2", "G3"]
    missing = [c for c in response_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing response columns: {missing}")

    y = df[response_cols].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    xdf = df.drop(columns=response_cols).copy()
    xdf = pd.get_dummies(xdf, dtype=float)
    xdf = xdf.replace([np.inf, -np.inf], np.nan)
    xdf = xdf.apply(pd.to_numeric, errors="coerce")
    xdf = xdf.fillna(xdf.median(numeric_only=True)).fillna(0.0)
    x = xdf.to_numpy(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Student data contain non-finite values after preprocessing")
    return x, y, list(xdf.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", choices=["aks_nnr", "full_dnn", "linear"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--d_max", type=int, default=10)
    ap.add_argument("--m", type=int, default=0)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--lambda_c", type=float, default=0.005)
    ap.add_argument("--lambda_n", type=float, default=None)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_split", type=float, default=0.30)
    ap.add_argument("--stop_threshold", type=float, default=0.30)
    ap.add_argument("--max_epochs", type=int, default=800)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_standardize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    x, y, predictor_names = load_student(args.data)
    row, ghat = base.run_one(x, y, args, args.seed)
    row.update({
        "dataset": "Student Performance Portuguese",
        "response_variables": "G1,G2,G3",
        "predictor_dim_after_encoding": int(x.shape[1]),
        "excluded_target_columns": "G1,G2,G3",
        "predictors_standardized_using_training_split": not args.no_standardize,
    })

    result_path = os.path.join(args.out_dir, f"results_seed{args.seed}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with open(os.path.join(args.out_dir, "results.csv"), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    if ghat is not None:
        np.savez_compressed(os.path.join(args.out_dir, f"ghat_seed{args.seed}.npz"), ghat_test=ghat)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
