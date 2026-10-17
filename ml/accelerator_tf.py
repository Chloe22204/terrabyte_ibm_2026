"""
TimeToTalk on the IBM z17 AI accelerator (NNPA / Telum II).

Same method as IBM's Digit_Class_TensorFlow lab: TensorFlow + IBM's `ibm-zdnn-plugin`, which sends
Dense / MatMul / ReLU / Sigmoid maths to the on-chip Neural Network Processing Assist (NNPA)
transparently.

What it does
  1. loads our MAIN model (the logistic regression from train.py) into TensorFlow as a one-layer network
     (a logistic regression is exactly Dense(1, sigmoid)), so the production model itself runs on the NNPA
  2. checks that the IBM zDNN plugin and its NNPA kernel library are loaded
  3. benchmarks scoring 10,000 patients: TensorFlow + NNPA vs the same weights in plain NumPy (CPU)
  4. saves artifacts/nnpa_logit.keras and artifacts/accelerator_benchmark.json

Run on LinuxONE from the repo root (after ml/train.py; ~1 GB free RAM, ~20 s):
  python ml/accelerator_tf.py
"""
import glob, importlib.metadata, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load, build_features, six_month_status, split_indices, SEED  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/support2.csv"
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

tf.keras.utils.set_random_seed(SEED)

# ---------------------------------------------------------------- 1. accelerator check (as in IBM's lab)
try:
    zdnn_version = importlib.metadata.version("ibm-zdnn-plugin")
except importlib.metadata.PackageNotFoundError:
    zdnn_version = None
z_ops = glob.glob(os.path.join(os.path.dirname(tf.__file__), "core", "kernels", "_z_ops.so"))
nnpa_flag = False
try:
    nnpa_flag = "nnpa" in open("/proc/cpuinfo").read().split()
except OSError:
    pass

# ---------------------------------------------------------------- 2. data: same features + split as train.py
df = load(DATA).reset_index(drop=True)
T, E = df["d.time"].to_numpy(float), df["death"].to_numpy(int)
tr, va, te = split_indices(T, E)
X, _ = build_features(df, tr)
feats = list(X.columns)
X = X.to_numpy(np.float32)
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
X = ((X - mu) / sd).astype(np.float32)
y = six_month_status(T, E)


def known(idx):
    k = ~np.isnan(y[idx])
    return X[idx][k], y[idx][k].astype(np.float32)


Xtr, ytr = known(tr)
Xva, yva = known(va)
Xte, yte = known(te)

# ---------------------------------------------------------------- 3. put OUR MAIN MODEL on the accelerator
# The main TimeToTalk model is an L2 logistic regression (artifacts/logit_model.json). A logistic regression
# is exactly a one-layer neural network: Dense(1, sigmoid). We load its trained weights into TensorFlow,
# so the production model itself runs on the NNPA, with the same answers as the API.
lm = json.load(open(f"{OUT}/logit_model.json"))
assert lm["feature_names"] == feats, "feature order differs from train.py - retrain first"
lm_mu, lm_sd = np.array(lm["mean"], np.float32), np.array(lm["std"], np.float32)
Xraw, _ = build_features(df, tr)
Xs = ((Xraw.to_numpy(np.float32) - lm_mu) / lm_sd).astype(np.float32)
model = keras.Sequential([keras.Input(shape=(Xs.shape[1],)), keras.layers.Dense(1, activation="sigmoid")])
model.layers[0].set_weights([np.array(lm["coef"], np.float32).reshape(-1, 1),
                             np.array([lm["intercept"]], np.float32)])
k_te = ~np.isnan(y[te])
Xte, yte = Xs[te][k_te], y[te][k_te]
p_te = model(Xte, training=False).numpy().ravel()
test_auc = roc_auc_score(yte, p_te)
train_s = 0.0
model.save(f"{OUT}/nnpa_logit.keras")

# ---------------------------------------------------------------- 4. benchmark: TF+NNPA vs plain NumPy CPU
N, REPEATS = 10_000, 20
batch = np.resize(Xte, (N, Xte.shape[1])).astype(np.float32)
tf_infer = tf.function(lambda x: model(x, training=False))
tf_infer(batch[:1]); tf_infer(batch)                       # warm-up / graph build


def median_time(fn):
    ts = []
    for _ in range(REPEATS):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


tf_s = median_time(lambda: tf_infer(batch).numpy())

W = model.get_weights()                                   # weights, bias
def numpy_infer(x):
    return 1 / (1 + np.exp(-(x @ W[0] + W[1])))

np_s = median_time(lambda: numpy_infer(batch))
same = float(np.max(np.abs(numpy_infer(batch[:500]).ravel() - tf_infer(batch[:500]).numpy().ravel())))

one = batch[:1]
tf_infer(one)
one_ms = median_time(lambda: tf_infer(one).numpy()) * 1000

res = {
    "ibm_zdnn_plugin": zdnn_version, "nnpa_kernel_library": z_ops[0] if z_ops else None,
    "cpu_nnpa_flag": nnpa_flag, "tensorflow": tf.__version__,
    "main_model_auc_6m_on_nnpa": round(float(test_auc), 4),
    "benchmark_patients": N,
    "tf_nnpa_ms_per_10k": round(tf_s * 1000, 2), "tf_nnpa_patients_per_s": round(N / tf_s),
    "numpy_cpu_ms_per_10k": round(np_s * 1000, 2), "numpy_cpu_patients_per_s": round(N / np_s),
    "speedup_vs_numpy": round(np_s / tf_s, 2), "single_patient_ms": round(one_ms, 3),
    "max_difference_tf_vs_numpy": same,
}
json.dump(res, open(f"{OUT}/accelerator_benchmark.json", "w"), indent=1)

line = "=" * 64
print(f"\n{line}\n TimeToTalk main model on the IBM z17 AI accelerator\n{line}")
print(f" machine                 : {os.uname().machine}   NNPA flag in /proc/cpuinfo: {nnpa_flag}")
print(f" ibm-zdnn-plugin         : {zdnn_version or 'NOT INSTALLED'}")
print(f" NNPA kernel library     : {'loaded (' + os.path.basename(z_ops[0]) + ')' if z_ops else 'NOT FOUND'}")
print(f" TensorFlow              : {tf.__version__}")
print(f" main model on NNPA, AUC : {test_auc:.3f}   (logistic regression, same weights as the API)")
print(f" score {N:,} patients     : TF+NNPA {tf_s*1000:7.1f} ms  ({N/tf_s:,.0f} patients/s)")
print(f"                           NumPy CPU {np_s*1000:5.1f} ms  ({N/np_s:,.0f} patients/s)")
print(f" speed-up vs NumPy       : {np_s/tf_s:.2f}x")
print(f" one patient (TF+NNPA)   : {one_ms:.2f} ms")
print(f" TF vs NumPy max diff    : {same:.2e}  (NNPA reduced precision)")
print(line)
