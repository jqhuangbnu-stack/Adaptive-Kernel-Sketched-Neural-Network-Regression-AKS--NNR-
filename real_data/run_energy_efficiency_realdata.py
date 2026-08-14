"""Energy Efficiency regression: AKS-NNR, Full-DNN, and linear baseline."""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Step2_model_training import KernelModelTrainer
from Step3_dimension_estimation import DimensionEstimator


def load_energy(path):
    # Read the simple UCI workbook without requiring pandas/openpyxl.
    # This is deliberately kept dependency-free for the cluster environment.
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("x:si", ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//x:t", ns)))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = []
        for row in sheet.findall(".//x:sheetData/x:row", ns):
            vals = {}
            for cell in row.findall("x:c", ns):
                ref = cell.attrib.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                col = 0
                for ch in letters:
                    col = col * 26 + ord(ch.upper()) - ord("A") + 1
                value = cell.find("x:v", ns)
                if value is None:
                    continue
                raw = value.text or ""
                if cell.attrib.get("t") == "s":
                    raw = shared[int(raw)]
                try:
                    vals[col - 1] = float(raw)
                except (TypeError, ValueError):
                    # The workbook has a header row such as X1, X2, ... .
                    vals[col - 1] = np.nan
            if vals and any(np.isfinite(v) for v in vals.values()):
                rows.append([vals.get(i, np.nan) for i in range(max(vals) + 1)])
    data = np.asarray(rows, dtype=np.float32)
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[1] < 10:
        raise ValueError(f"Expected at least 10 numeric columns, got {data.shape}")
    return data[:, :8], data[:, -2:]


def split_data(n, seed, test_frac):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt = max(1, int(round(test_frac * n)))
    return idx[nt:], idx[:nt]


def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def median_bandwidth(X, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(200000, n * max(n - 1, 1) // 2)
    i = rng.integers(0, n, k)
    j = rng.integers(0, n, k)
    keep = i != j
    sq = np.sum((X[i[keep]] - X[j[keep]]) ** 2, axis=1)
    sq = sq[sq > 1e-10]
    return float(np.sqrt(np.median(sq) / 2.0)) if len(sq) else 1.0


class Sketcher(nn.Module):
    def __init__(self, m, n, h, device, seed):
        super().__init__()
        self.m, self.n, self.h = m, n, float(h)
        self.device = torch.device(device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        self.register_buffer("S", torch.randn(m, n, generator=gen, device=self.device))
        self.register_buffer("sqrt_n", torch.sqrt(torch.tensor(float(n), device=self.device)))

    def compute_initial_loss(self, Y):
        return torch.mean((Y - Y.mean(0, keepdim=True)) ** 2)


def kernel_features(Xeval, Xtrain, sketcher):
    xe = torch.as_tensor(Xeval, dtype=torch.float32, device=sketcher.device)
    xt = torch.as_tensor(Xtrain, dtype=torch.float32, device=sketcher.device)
    tn = torch.sum(xt * xt, dim=1)
    out = []
    for start in range(0, len(xe), 256):
        x = xe[start:start + 256]
        sq = (torch.sum(x * x, 1, keepdim=True) + tn[None, :] - 2 * (x @ xt.T)).clamp_min(0)
        k = torch.exp(-sq / (2 * sketcher.h ** 2)) / sketcher.sqrt_n
        # SketchedModel expects the convention (m, n_eval).
        out.append(sketcher.S @ k.T)
    return torch.cat(out, 1)


def fit_full_dnn(Xtr, Ytr, args, seed, device):
    torch.manual_seed(seed)
    model_layers = []
    cur = Xtr.shape[1]
    for _ in range(max(0, args.L - 1)):
        model_layers += [nn.Linear(cur, args.hidden_dim), nn.ReLU()]
        cur = args.hidden_dim
    model_layers.append(nn.Linear(cur, Ytr.shape[1]))
    model = nn.Sequential(*model_layers).to(device)
    rng = np.random.default_rng(seed + 7919)
    idx = rng.permutation(len(Xtr))
    nv = max(1, int(round(args.val_split * len(Xtr))))
    vi, ti = idx[:nv], idx[nv:]
    xt = torch.tensor(Xtr[ti], dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr[ti], dtype=torch.float32, device=device)
    xv = torch.tensor(Xtr[vi], dtype=torch.float32, device=device)
    yv = torch.tensor(Ytr[vi], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best, state, wait = float("inf"), None, 0
    for _ in range(args.max_epochs):
        model.train(); opt.zero_grad()
        loss_fn(model(xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad(): val = loss_fn(model(xv), yv).item()
        if val < best - 1e-8:
            best, state, wait = val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience: break
    if state is not None: model.load_state_dict(state)
    return model.eval()


def gmdd_matrix(g_hat, y):
    """Section 5 empirical GMDD matrix used for order estimation."""
    g_hat = np.asarray(g_hat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    centered = g_hat - g_hat.mean(axis=0, keepdims=True)
    diff = y[:, None, :] - y[None, :, :]
    dist_y = np.sqrt(np.sum(diff * diff, axis=2))
    upper_dist = np.triu(dist_y, k=1)
    delta = -(centered.T @ upper_dist @ centered)
    delta /= g_hat.shape[0] * (g_hat.shape[0] - 1) / 2.0
    return 0.5 * (delta + delta.T)


def choose_gmdd_dimension(g_hat, y, rho=0.90):
    delta = gmdd_matrix(g_hat, y)
    eigenvalues, eigenvectors = np.linalg.eigh(delta)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = float(eigenvalues.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return 1, eigenvalues, np.zeros_like(eigenvalues), True
    cumulative = np.cumsum(eigenvalues) / total
    d_hat = int(np.searchsorted(cumulative, rho, side="left") + 1)
    return d_hat, eigenvalues, cumulative, False


def run_one(X, Y, args, seed):
    started = time.time()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    tr, te = split_data(len(X), seed, args.test_frac)
    Xtr, Xte = standardize(X[tr], X[te]) if not args.no_standardize else (X[tr], X[te])
    Ytr, Yte = Y[tr], Y[te]
    result = {"seed": seed, "n": len(X), "p": X.shape[1], "n_train": len(tr), "n_test": len(te),
              "test_mse": None, "d_estimated": None, "fallback_used": None,
              "runtime_seconds": None, "method": args.method, "status": "success"}
    if args.method == "linear":
        xt = np.column_stack([np.ones(len(Xtr)), Xtr])
        coef = np.linalg.lstsq(xt, Ytr, rcond=None)[0]
        pred = np.column_stack([np.ones(len(Xte)), Xte]) @ coef
    elif args.method == "full_dnn":
        model = fit_full_dnn(Xtr, Ytr, args, seed, device)
        with torch.no_grad(): pred = model(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    elif args.method == "aks_nnr":
        h = args.h if args.h > 0 else median_bandwidth(Xtr, seed)
        m = args.m if args.m > 0 else len(Xtr)
        sk = Sketcher(m, len(Xtr), h, device, seed)
        sk.X_train = torch.tensor(Xtr, dtype=torch.float32, device=device)
        sk.X_train_norm = torch.sum(sk.X_train ** 2, dim=1)
        yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
        trainer = KernelModelTrainer(sketcher=sk, lambda_c=args.lambda_c, lr=args.lr, L=args.L,
            hidden_dim=args.hidden_dim, patience=args.patience, val_split=args.val_split,
            stop_threshold=args.stop_threshold, max_epochs=args.max_epochs)
        trained, final_d = trainer.run_iterative_training(sk.X_train, yt, d_max=args.d_max)
        lam = args.lambda_n if args.lambda_n is not None else len(Xtr) ** (-2 / (2 + args.d_max)) * np.log(np.log(len(Xtr)))
        est = DimensionEstimator(trainer.history_losses, len(Xtr), final_d, lambda_n=lam)
        dhat = max(1, min(int(est.estimate_dimension()), max(trained.keys())))
        model = trained[dhat].eval()
        feat = kernel_features(Xte, Xtr, sk)
        with torch.no_grad(): pred = model(feat).cpu().numpy(); ghat = model.transform(feat).cpu().numpy()
        result.update({"d_estimated": dhat, "fallback_used": bool(est.dimension_fallback_used), "h": h,
                       "m_sketch": m, "lambda_n": float(lam), "r_hats": json.dumps(est.r_hats)})
    elif args.method in {"gmddnet_s", "gmddnet_f"}:
        nsdr_dir = os.environ.get("NSDR_DIR", os.path.join(THIS_DIR, "aos_code"))
        sys.path.insert(0, nsdr_dir)
        from NSDR import NSDR
        # Compatibility for newer scikit-learn versions used on the server.
        if not hasattr(NSDR, "_validate_data"):
            from sklearn.utils import check_X_y, check_array
            def _validate_data(self, X, y=None, **kwargs):
                kwargs.pop("reset", None)
                return check_array(X, **kwargs) if y is None else check_X_y(X, y, **kwargs)
            NSDR._validate_data = _validate_data
        nsdr_device = "cuda:0" if device == "cuda" else "cpu"
        from sklearn.linear_model import LinearRegression
        if args.method == "gmddnet_s":
            # The successive method extracts one direction at each step.
            # Leave the network unspecified so NSDR creates a one-output
            # network per step, while the full multivariate response is used
            # in the GMDD objective.
            model = NSDR(max_dim=args.gmdd_max_dim,
                method="seq", iter_num=args.iter_num, batch_size=args.batch_size,
                lr=args.lr, retrain=True, adaptive_cv=False, y_mode=0,
                device=nsdr_device, random_state=seed)
            model.fit(Xtr, Ytr)
            ztr_full = np.asarray(model.transform(Xtr), dtype=np.float64)
            zte_full = np.asarray(model.transform(Xte), dtype=np.float64)
            dhat, eig, cumulative, order_fallback = choose_gmdd_dimension(
                ztr_full, Ytr, rho=args.gmdd_rho
            )
            dhat = max(1, min(dhat, ztr_full.shape[1]))
            Ztr, Zte = ztr_full[:, :dhat], zte_full[:, :dhat]
            rep_dim = Zte.shape[1]
        else:
            net = NSDR.generate_default_net(Xtr.shape[1], output_dim=args.gmdd_max_dim)
            model = NSDR(neural_network=net, max_dim=1, method="fnorm",
                iter_num=args.iter_num, batch_size=args.batch_size, lr=args.lr,
                retrain=True, adaptive_cv=False, y_mode=0, device=nsdr_device,
                random_state=seed)
            model.fit(Xtr, Ytr)
            ztr_full = np.asarray(model.transform(Xtr), dtype=np.float64)
            zte_full = np.asarray(model.transform(Xte), dtype=np.float64)
            dhat, eig, cumulative, order_fallback = choose_gmdd_dimension(
                ztr_full, Ytr, rho=args.gmdd_rho
            )
            dhat = max(1, min(dhat, ztr_full.shape[1]))
            Ztr, Zte = ztr_full[:, :dhat], zte_full[:, :dhat]
            rep_dim = Zte.shape[1]
        # Evaluate the learned sufficient predictors with two downstream
        # regressors: a linear map and a DNN map.
        reg = LinearRegression().fit(Ztr, Ytr)
        pred_linear = reg.predict(Zte)
        downstream_dnn = fit_full_dnn(
            Ztr, Ytr, args, seed + 104729, device
        )
        with torch.no_grad():
            pred_dnn = downstream_dnn(
                torch.tensor(Zte, dtype=torch.float32, device=device)
            ).cpu().numpy()
        mse_linear = float(np.mean((pred_linear - Yte) ** 2))
        mse_dnn = float(np.mean((pred_dnn - Yte) ** 2))
        pred = pred_linear
        result.update({"representation_dim": int(rep_dim),
                       "fallback_used": bool(order_fallback),
                       "d_estimated": int(dhat),
                       "gmdd_rho": float(args.gmdd_rho),
                       "test_mse_linear": mse_linear,
                       "test_mse_dnn": mse_dnn})
        result.update({"eigenvalues": eig.tolist(),
                       "cumulative_ratio": cumulative.tolist()})
    else:
        raise ValueError(args.method)
    result["test_mse"] = float(np.mean((pred - Yte) ** 2))
    result["runtime_seconds"] = time.time() - started
    return result, (ghat if args.method == "aks_nnr" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", choices=["aks_nnr", "full_dnn", "linear", "gmddnet_s", "gmddnet_f"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--test_frac", type=float, default=.30); ap.add_argument("--d_max", type=int, default=10)
    ap.add_argument("--m", type=int, default=0); ap.add_argument("--h", type=float, default=0.)
    ap.add_argument("--lambda_c", type=float, default=.005); ap.add_argument("--lambda_n", type=float, default=None)
    ap.add_argument("--lr", type=float, default=.001); ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=256); ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_split", type=float, default=.30); ap.add_argument("--stop_threshold", type=float, default=.30)
    ap.add_argument("--max_epochs", type=int, default=800); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gmdd_max_dim", type=int, default=10)
    ap.add_argument("--gmdd_rho", type=float, default=0.90)
    ap.add_argument("--iter_num", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--no_standardize", action="store_true"); args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    X, Y = load_energy(args.data)
    row, ghat = run_one(X, Y, args, args.seed)
    with open(os.path.join(args.out_dir, f"results_seed{args.seed}.json"), "w") as f: json.dump(row, f, indent=2)
    with open(os.path.join(args.out_dir, "results.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0: w.writeheader()
        w.writerow(row)
    if ghat is not None: np.savez_compressed(os.path.join(args.out_dir, f"ghat_seed{args.seed}.npz"), ghat_test=ghat)
    print(json.dumps(row), flush=True)


if __name__ == "__main__": main()
