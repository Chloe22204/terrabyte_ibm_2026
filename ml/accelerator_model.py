"""
Second-opinion neural net for the IBM z17 AI accelerator (Telum / NNPA).
UNTESTED on LinuxONE - adapt using the IBM lab notebooks (fraud LSTM / MNIST CNN), which
show exactly how they export to ONNX and run on the accelerator. Copy their inference cell.

What it does:
 1. trains a small PyTorch MLP for "died within 6 months" on the SAME features as the Cox model
 2. exports it to ONNX (portable format the labs use for the accelerator)
 3. benchmarks batch inference -> numbers for the video ("scores N patients per second")

Run after train.py:  python ml/accelerator_model.py --data data/support2.csv --out artifacts
"""
import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from train import load, build_features, six_month_status, SEED


def main(a):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = load(a.data).reset_index(drop=True)
    T, E = df["d.time"].to_numpy(float), df["death"].to_numpy(int)
    perm = rng.permutation(len(df))                 # same split as train.py (same seed, same order)
    n_tr, n_va = int(0.6 * len(df)), int(0.2 * len(df))
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    X, _ = build_features(df, tr)
    X = X.to_numpy(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd
    y = six_month_status(T, E)

    def xy(idx):
        k = ~np.isnan(y[idx])
        return torch.tensor(X[idx][k]), torch.tensor(y[idx][k], dtype=torch.float32)

    Xtr, ytr = xy(tr); Xva, yva = xy(va); Xte, yte = xy(te)
    model = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = 0, None
    for epoch in range(60):
        model.train()
        for i in torch.randperm(len(Xtr)).split(256):
            opt.zero_grad(); lossf(model(Xtr[i]).squeeze(1), ytr[i]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(model(Xva)).squeeze(1))
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        test_auc = roc_auc_score(yte, torch.sigmoid(model(Xte)).squeeze(1))
    print(f"MLP val AUC {best:.3f}  test AUC {test_auc:.3f}")

    # export to ONNX (sigmoid included so output = probability)
    export = nn.Sequential(model, nn.Sigmoid())
    torch.onnx.export(export, torch.zeros(1, X.shape[1]), f"{a.out}/mlp_6m.onnx",
                      input_names=["x"], output_names=["p"], dynamic_axes={"x": {0: "batch"}})
    json.dump({"mean": mu.tolist(), "std": sd.tolist()}, open(f"{a.out}/mlp_scaler.json", "w"))

    # benchmark (PyTorch CPU path; then repeat with the lab's accelerator-enabled runtime)
    batch = torch.tensor(np.repeat(X[te], max(1, 10000 // len(te) + 1), axis=0)[:10000])
    with torch.no_grad():
        model(batch)                                  # warm-up
        t0 = time.perf_counter(); model(batch); dt = time.perf_counter() - t0
    res = {"mlp_test_auc": float(test_auc), "benchmark_patients": 10000,
           "seconds": dt, "patients_per_second": 10000 / dt}
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(f"{a.out}/mlp_6m.onnx")
        s.run(None, {"x": batch.numpy()})
        t0 = time.perf_counter(); s.run(None, {"x": batch.numpy()}); dt2 = time.perf_counter() - t0
        res["onnxruntime_patients_per_second"] = 10000 / dt2
    except ImportError:
        pass
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{a.out}/accelerator_benchmark.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/support2.csv")
    p.add_argument("--out", default="artifacts")
    main(p.parse_args())
