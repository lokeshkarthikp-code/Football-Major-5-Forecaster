"""
Calibration: learning from how past forecasts turned out.

WHAT THIS IS, AND WHAT REFITTING IS NOT
---------------------------------------
The weekly refit re-estimates team strength from newer goals. That is
learning about TEAMS. It is not learning about the MODEL: nothing in the
pipeline notices that the model's 55% calls only come in 51% of the time
and corrects for it.

This module does that. It takes a history of (predicted probability,
what actually happened) and fits a small correction that maps raw model
probabilities to calibrated ones. The Dixon-Coles fit is untouched --
this sits downstream of it, adjusting outputs only.

WHY THE CORRECTION IS TINY
--------------------------
Two methods, both deliberately low-capacity:

    temperature   one parameter.  p ~ p^(1/T)
    vector        six parameters. p ~ exp(a_k * log p_k + b_k)

Temperature sharpens or softens the whole distribution: T > 1 means the
model was overconfident and gets pulled toward the base rate, T < 1 means
underconfident. Vector scaling additionally lets each outcome shift on its
own, which is what you want if draws specifically are mispriced.

A richer calibrator (isotonic, splines, per-bin lookups) would fit the
noise in a few thousand football matches and make things worse out of
sample. The measured calibration table already tracks the diagonal
closely, so there is very little signal here to extract. One to six
parameters is the honest budget.

THE LEAK YOU MUST AVOID
-----------------------
Fitting the calibrator on all predictions and then reporting improved
metrics on those same predictions is circular, and it will always look
like it worked. So `walk_forward_calibration` refits the calibrator using
only predictions whose matches were played strictly before each test
block. That is the same discipline the model backtest already applies,
for the same reason.

THE RESULT MAY WELL BE "NO IMPROVEMENT"
---------------------------------------
That is a finding, not a failure. `compare` reports raw against calibrated
on identical matches and will say so plainly. A calibration layer that
does not help means the model was already well calibrated, which is worth
knowing and worth saying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .backtest import OUTCOMES, brier, log_loss

EPS = 1e-12


def _as_array(probs: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(probs, pd.DataFrame):
        p = probs[OUTCOMES].to_numpy(dtype=float)
    else:
        p = np.asarray(probs, dtype=float)
    p = np.clip(p, EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _onehot(actual: pd.Series | np.ndarray) -> np.ndarray:
    a = pd.Series(actual).to_numpy()
    return np.stack([(a == oc).astype(float) for oc in OUTCOMES], axis=1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------

@dataclass
class Calibrator:
    """A fitted probability correction.

    method
        "temperature" (1 parameter) or "vector" (6). Temperature is the
        safer default; vector is worth trying only once you have a few
        thousand settled predictions.
    """

    method: str = "temperature"
    params: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    n_train: int = 0
    fitted: bool = False

    # -- fitting ------------------------------------------------------------

    def fit(self, probs, actual) -> "Calibrator":
        p = _as_array(probs)
        y = _onehot(actual)
        logp = np.log(p)
        self.n_train = len(p)

        if self.method == "temperature":
            def nll(t):
                return -np.sum(y * np.log(_softmax(logp / t[0]) + EPS)) / len(p)

            res = minimize(nll, np.array([1.0]), method="L-BFGS-B",
                           bounds=[(0.25, 4.0)])
            self.params = res.x

        elif self.method == "vector":
            def nll(w):
                a, b = w[:3], w[3:]
                return -np.sum(y * np.log(_softmax(a * logp + b) + EPS)) / len(p)

            x0 = np.concatenate([np.ones(3), np.zeros(3)])
            res = minimize(nll, x0, method="L-BFGS-B",
                           bounds=[(0.25, 4.0)] * 3 + [(-2.0, 2.0)] * 3)
            self.params = res.x

        else:
            raise ValueError(f"Unknown method {self.method!r}. "
                             "Use 'temperature' or 'vector'.")

        self.fitted = True
        return self

    # -- applying -----------------------------------------------------------

    def apply(self, probs) -> np.ndarray:
        p = _as_array(probs)
        if not self.fitted:
            return p
        logp = np.log(p)
        if self.method == "temperature":
            return _softmax(logp / self.params[0])
        a, b = self.params[:3], self.params[3:]
        return _softmax(a * logp + b)

    def apply_frame(self, probs: pd.DataFrame) -> pd.DataFrame:
        out = probs.copy()
        out[OUTCOMES] = self.apply(probs)
        return out

    # -- interpretation -----------------------------------------------------

    def describe(self) -> str:
        if not self.fitted:
            return "unfitted (pass-through)"
        if self.method == "temperature":
            t = float(self.params[0])
            direction = ("overconfident, pulled toward the base rate"
                         if t > 1.02 else
                         "underconfident, sharpened" if t < 0.98 else
                         "already well calibrated, near no-op")
            return f"temperature T={t:.3f} ({direction}), n={self.n_train}"
        a, b = self.params[:3], self.params[3:]
        parts = ", ".join(f"{oc}: a={ai:.2f} b={bi:+.2f}"
                          for oc, ai, bi in zip(OUTCOMES, a, b))
        return f"vector [{parts}], n={self.n_train}"


# ---------------------------------------------------------------------------
# Honest evaluation
# ---------------------------------------------------------------------------

def expected_calibration_error(probs, actual, bins: int = 10) -> float:
    """Mean gap between predicted and observed frequency, pooled over
    outcomes and weighted by bin size. Lower is better; 0 is perfect."""
    p = _as_array(probs).ravel()
    y = _onehot(actual).ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total, err = len(p), 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += m.sum() / total * abs(p[m].mean() - y[m].mean())
    return float(err)


def walk_forward_calibration(results: pd.DataFrame, *,
                             method: str = "temperature",
                             min_train: int = 400,
                             step_days: int = 28,
                             league: str | None = None) -> pd.DataFrame:
    """Apply calibration the only way that can be honestly evaluated.

    For each time block, the calibrator is fitted on predictions whose
    matches finished strictly before that block begins, then applied to
    the block. No prediction ever contributes to the correction applied
    to itself.

    Returns the input rows that were calibrated, with the raw
    probabilities kept alongside the corrected ones so the two can be
    scored on identical matches.
    """
    df = results.copy()
    if league is not None and "league" in df.columns:
        df = df[df["league"] == league]
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) <= min_train:
        raise ValueError(
            f"Need > {min_train} predictions to calibrate; got {len(df)}."
        )

    out_rows = []
    start = df.loc[min_train, "date"]
    cursor = start
    end = df["date"].max()

    while cursor <= end:
        nxt = cursor + pd.Timedelta(days=step_days)
        block = df[(df["date"] >= cursor) & (df["date"] < nxt)]
        train = df[df["date"] < cursor]

        if block.empty:
            cursor = nxt
            continue
        if len(train) < min_train:
            cursor = nxt
            continue

        cal = Calibrator(method=method).fit(train, train["actual"])
        adj = cal.apply(block)

        b = block.copy()
        for i, oc in enumerate(OUTCOMES):
            b[f"raw_{oc}"] = block[oc].to_numpy()
            b[f"cal_{oc}"] = adj[:, i]
        b["cal_param"] = (cal.params[0] if method == "temperature"
                          else np.nan)
        b["cal_n_train"] = cal.n_train
        out_rows.append(b)
        cursor = nxt

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def compare(calibrated: pd.DataFrame) -> pd.DataFrame:
    """Score raw against calibrated on exactly the same matches."""
    if calibrated.empty:
        return pd.DataFrame()

    actual = calibrated["actual"]
    raw = calibrated[[f"raw_{oc}" for oc in OUTCOMES]]
    raw.columns = OUTCOMES
    cal = calibrated[[f"cal_{oc}" for oc in OUTCOMES]]
    cal.columns = OUTCOMES

    rows = []
    for label, p in (("Raw model", raw), ("Calibrated", cal)):
        rows.append({
            "version": label,
            "brier": brier(p, actual),
            "log_loss": log_loss(p, actual),
            "ece": expected_calibration_error(p, actual),
            "accuracy": float(
                (p[OUTCOMES].idxmax(axis=1).to_numpy() == actual.to_numpy()).mean()
            ),
            "n": len(p),
        })
    out = pd.DataFrame(rows)

    d = out.set_index("version")

    # Paired standard error on the Brier difference. Raw and calibrated are
    # scored on the SAME matches, so the per-match differences are paired and
    # their spread is the right yardstick -- far tighter, and far more
    # honest, than treating the two Brier scores as independent samples.
    y = _onehot(actual)
    per_raw = ((_as_array(raw) - y) ** 2).sum(axis=1)
    per_cal = ((_as_array(cal) - y) ** 2).sum(axis=1)
    diff = per_cal - per_raw
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")

    delta = {
        "version": "Change",
        "brier": d.loc["Calibrated", "brier"] - d.loc["Raw model", "brier"],
        "log_loss": d.loc["Calibrated", "log_loss"] - d.loc["Raw model", "log_loss"],
        "ece": d.loc["Calibrated", "ece"] - d.loc["Raw model", "ece"],
        "accuracy": d.loc["Calibrated", "accuracy"] - d.loc["Raw model", "accuracy"],
        "n": len(raw),
        "brier_diff_se": se,
    }
    return pd.concat([out, pd.DataFrame([delta])], ignore_index=True)


def verdict(comparison: pd.DataFrame) -> str:
    """Plain-language read on whether the layer earned its keep."""
    if comparison.empty:
        return "No calibrated predictions to compare."
    d = comparison.set_index("version")
    db = float(d.loc["Change", "brier"])
    n = int(d.loc["Change", "n"])
    se = float(d.loc["Change", "brier_diff_se"])

    # Two paired standard errors. Anything inside that band is a difference
    # the sample cannot distinguish from chance, and calling it a result
    # either way would be overclaiming.
    band = 2.0 * se

    if db < -band:
        return (f"Calibration helps: Brier improved by {abs(db):.4f} "
                f"(+/- {se:.4f} SE) over {n} out-of-sample predictions. "
                "Worth keeping.")
    if db > band:
        return (f"Calibration HURTS: Brier worsened by {db:.4f} "
                f"(+/- {se:.4f} SE) over {n} predictions. Do not deploy it -- "
                "the raw model is better.")
    return (f"No meaningful difference ({db:+.4f} Brier, SE {se:.4f}, over "
            f"{n} predictions -- inside two standard errors). The model was "
            "already well calibrated, which is the honest conclusion: there "
            "was little here for this layer to correct.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from .multileague import load_backtest

    ap = argparse.ArgumentParser(
        description="Fit and honestly evaluate a calibration layer.")
    ap.add_argument("--league", default="ENG-Premier League")
    ap.add_argument("--method", default="temperature",
                    choices=["temperature", "vector"])
    ap.add_argument("--min-train", type=int, default=400)
    ap.add_argument("--step-days", type=int, default=28)
    args = ap.parse_args()

    res = load_backtest()
    print(f"\nCalibration check: {args.league}, method={args.method}")
    print("Walk-forward -- the calibrator never sees the matches it corrects.\n")

    cal = walk_forward_calibration(
        res, method=args.method, min_train=args.min_train,
        step_days=args.step_days, league=args.league,
    )
    if cal.empty:
        print("Not enough predictions to calibrate.")
        return

    cmp = compare(cal)
    print(cmp.round(4).to_string(index=False))

    if args.method == "temperature":
        t = cal["cal_param"].dropna()
        if len(t):
            print(f"\nTemperature over time: mean {t.mean():.3f}, "
                  f"range {t.min():.3f}-{t.max():.3f}")
            print("T > 1 = model was overconfident; T ~ 1 = already calibrated.")

    print("\n" + verdict(cmp))


if __name__ == "__main__":
    _main()