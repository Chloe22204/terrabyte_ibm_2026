"""
L2-regularised logistic regression for "died within 6 months", saved as plain JSON.

Why this is the main model: SUPPORT2 follows every patient to death or past 6 months, so a
direct 6-month classifier loses nothing to censoring, and every factor is explainable. The Cox
model scores about the same (test AUC within noise), so it is kept only for the survival-curve
shape. Fitting uses scikit-learn; prediction needs only numpy, so the
API container stays small and there is no pickle.
"""
import json
import numpy as np


class LogisticModel:
    def __init__(self, C=0.5):
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.feature_names_ = None
        self.mean_ = None
        self.std_ = None

    def _standardize(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit(self, X, y, feature_names=None):
        from sklearn.linear_model import LogisticRegression    # training only; the API never imports it
        X = np.asarray(X, dtype=float)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        clf = LogisticRegression(C=self.C, max_iter=3000).fit(self._standardize(X), y)
        self.coef_, self.intercept_ = clf.coef_[0], float(clf.intercept_[0])
        return self

    def logit(self, X):
        return self._standardize(X) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Probability of dying within 6 months, shape (n_samples,)."""
        return 1 / (1 + np.exp(-self.logit(X)))

    def contributions(self, x_row):
        """Per-feature contribution to the log-odds for one patient
        (coefficient x standardized value). Positive = pushes risk up."""
        z = self._standardize(np.asarray(x_row, dtype=float).reshape(1, -1))[0]
        return dict(zip(self.feature_names_, z * self.coef_))

    # ---------- persistence (JSON, no pickle) ----------
    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"C": self.C, "feature_names": self.feature_names_, "coef": self.coef_.tolist(),
                       "intercept": self.intercept_, "mean": self.mean_.tolist(), "std": self.std_.tolist()}, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["C"])
        m.feature_names_, m.intercept_ = d["feature_names"], d["intercept"]
        m.coef_, m.mean_, m.std_ = np.array(d["coef"]), np.array(d["mean"]), np.array(d["std"])
        return m


def anchor_curve(S, S_180, risk_6m):
    """Rescale Cox survival curves so each passes through 1 - risk_6m at day 180.
    Keeps the Cox curve's shape over time but makes it agree with the main 6-month risk.
    Under Cox, S(t) = exp(-c * H0(t)), so changing c gives S'(t) = S(t) ** (log(1 - risk) / log(S(180))).
    S: (n, n_times) Cox survival; S_180: (n,) Cox survival at day 180; risk_6m: (n,) main model risk."""
    S = np.clip(np.asarray(S, float), 1e-12, 1.0)
    S_180 = np.clip(np.asarray(S_180, float), 1e-12, 1 - 1e-12)
    power = np.log(np.clip(1 - np.asarray(risk_6m, float), 1e-12, 1 - 1e-12)) / np.log(S_180)
    return S ** power[:, None]
