"""
Minimal Cox proportional hazards model using only numpy + scipy.

Why this exists: lifelines depends on autograd-gamma, which can fail to build on some
platforms (including s390x). This implementation has no compiled dependencies beyond
numpy/scipy, so it runs anywhere the LinuxONE Jupyter image runs.

- Breslow partial likelihood with L2 (ridge) penalty, optimised with L-BFGS-B
- Breslow baseline cumulative hazard -> survival curves S(t | x)
- Harrell's concordance index
- Model can be saved to / loaded from plain JSON (no pickle), so the API container
  only needs numpy.
"""
import json
import numpy as np
from scipy.optimize import minimize


class CoxPH:
    def __init__(self, penalizer=0.1):
        self.penalizer = penalizer
        self.coef_ = None
        self.feature_names_ = None
        self.mean_ = None
        self.std_ = None
        self.base_times_ = None
        self.base_cumhaz_ = None

    # ---------- fitting ----------
    def _standardize(self, X):
        return (X - self.mean_) / self.std_

    def fit(self, X, durations, events, feature_names=None):
        X = np.asarray(X, dtype=float)
        T = np.asarray(durations, dtype=float)
        E = np.asarray(events, dtype=float)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Z = self._standardize(X)

        # sort by descending time so risk sets are cumulative sums
        order = np.argsort(-T, kind="mergesort")
        Z, T, E = Z[order], T[order], E[order]
        n, p = Z.shape
        lam = self.penalizer

        def negloglik(beta):
            eta = Z @ beta
            eta_max = eta.max()
            w = np.exp(eta - eta_max)
            cum_w = np.cumsum(w)                      # risk set sums (Breslow; ties handled approximately)
            cum_wZ = np.cumsum(w[:, None] * Z, axis=0)
            log_risk = np.log(cum_w) + eta_max
            ll = np.sum(E * (eta - log_risk))
            grad = (E[:, None] * (Z - cum_wZ / cum_w[:, None])).sum(axis=0)
            # ridge penalty
            ll -= 0.5 * lam * beta @ beta
            grad -= lam * beta
            return -ll / n, -grad / n

        res = minimize(negloglik, np.zeros(p), jac=True, method="L-BFGS-B")
        self.coef_ = res.x
        self._fit_baseline(Z, T, E)
        return self

    def _fit_baseline(self, Z, T, E):
        """Breslow estimator of the baseline cumulative hazard."""
        w = np.exp(Z @ self.coef_)
        event_times = np.unique(T[E == 1])
        cumhaz = []
        h = 0.0
        for t in event_times:
            d = np.sum((T == t) & (E == 1))
            r = np.sum(w[T >= t])
            h += d / r
            cumhaz.append(h)
        self.base_times_ = event_times
        self.base_cumhaz_ = np.array(cumhaz)

    # ---------- prediction ----------
    def linear_predictor(self, X):
        return self._standardize(np.asarray(X, dtype=float)) @ self.coef_

    def baseline_cumhaz_at(self, times):
        times = np.atleast_1d(times)
        idx = np.searchsorted(self.base_times_, times, side="right") - 1
        out = np.where(idx >= 0, self.base_cumhaz_[np.clip(idx, 0, None)], 0.0)
        return out

    def predict_survival(self, X, times):
        """Returns array (n_samples, n_times) of S(t | x)."""
        lp = self.linear_predictor(X)
        H0 = self.baseline_cumhaz_at(times)
        return np.exp(-np.exp(lp)[:, None] * H0[None, :])

    def contributions(self, x_row):
        """Per-feature contribution to the log-hazard for one patient
        (coefficient x standardized value). Positive = pushes risk up."""
        z = self._standardize(np.asarray(x_row, dtype=float).reshape(1, -1))[0]
        return dict(zip(self.feature_names_, z * self.coef_))

    # ---------- persistence (JSON, no pickle) ----------
    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({
                "penalizer": self.penalizer,
                "feature_names": self.feature_names_,
                "coef": self.coef_.tolist(),
                "mean": self.mean_.tolist(),
                "std": self.std_.tolist(),
                "base_times": self.base_times_.tolist(),
                "base_cumhaz": self.base_cumhaz_.tolist(),
            }, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(d["penalizer"])
        m.feature_names_ = d["feature_names"]
        for k_src, k_dst in [("coef", "coef_"), ("mean", "mean_"), ("std", "std_"),
                             ("base_times", "base_times_"), ("base_cumhaz", "base_cumhaz_")]:
            setattr(m, k_dst, np.array(d[k_src]))
        return m


def concordance_index(durations, risk_scores, events):
    """Harrell's C: fraction of comparable pairs where the higher-risk patient died first.
    Vectorised in chunks; fine for ~10k patients."""
    T = np.asarray(durations, float)
    R = np.asarray(risk_scores, float)
    E = np.asarray(events, bool)
    num, den = 0.0, 0.0
    idx = np.where(E)[0]
    for start in range(0, len(idx), 500):
        i = idx[start:start + 500]
        comparable = T[None, :] > T[i][:, None]           # j outlived i, and i had the event
        diff = R[i][:, None] - R[None, :]
        num += np.sum(comparable & (diff > 0)) + 0.5 * np.sum(comparable & (diff == 0))
        den += np.sum(comparable)
    return num / den if den else np.nan


def kaplan_meier(durations, events, t):
    """Kaplan-Meier survival estimate at time t."""
    T = np.asarray(durations, float)
    E = np.asarray(events, bool)
    s = 1.0
    for u in np.unique(T[E & (T <= t)]):
        at_risk = np.sum(T >= u)
        d = np.sum((T == u) & E)
        s *= 1 - d / at_risk
    return s
