# -*- coding: utf-8 -*-
"""
Generates Figure 5 (selective risk-coverage curves on JNU) for the
manuscript. Persisted as its own script (2026-09-17, external review) so
the figure is reproducible from a named entry point rather than an ad hoc
command; reuses the same JNU leave-1000rpm-out split, scaler-then-SMOTE
order, and RF+XGB+LR ensemble as pipeline_v3_grouped.py and
baseline_comparison_jnu.py, and the same tie-corrected risk-coverage
computation as baseline_comparison_jnu.py.aurc() (expected risk under
uniform-random tie-breaking, not raw argsort order).
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

from pipeline_v3_grouped import build_jnu_grouped

OUT = os.path.dirname(os.path.abspath(__file__))


def risk_coverage_tiecorrected(scores, errs):
    """Same tie-handling as baseline_comparison_jnu.aurc(): risk within a
    tie group is its expected value under uniform-random tie-breaking."""
    order = np.argsort(scores, kind="stable")
    s_sorted = np.asarray(scores)[order]
    e_sorted = np.asarray(errs)[order]
    n = len(e_sorted)
    cum_err_expected = np.empty(n, dtype=float)
    before_err = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        m = j - i
        e = e_sorted[i:j].sum()
        for k in range(1, m + 1):
            cum_err_expected[i + k - 1] = before_err + k * e / m
        before_err += e
        i = j
    coverage = np.arange(1, n + 1) / n
    risk = cum_err_expected / np.arange(1, n + 1)
    return coverage, risk


if __name__ == "__main__":
    print("Building JNU (grouped, leave-1000rpm-out)...")
    Xtr, ytr, Xte, yte = build_jnu_grouped()

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(Xtr_sc, ytr)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42,
                         use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys); xgb.fit(Xs_sc, ys); lr.fit(Xs_sc, ys)

    P_rf = rf.predict_proba(Xte_sc); P_xgb = xgb.predict_proba(Xte_sc); P_lr = lr.predict_proba(Xte_sc)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != yte).astype(int)
    print(f"Ensemble accuracy check: {(1-errors.mean())*100:.2f}%")

    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0
    errs_lr = (np.argmax(P_lr, axis=1) != yte).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, score, errs, style, lw in [
        ("I1_hat (predictive entropy)", entropy, errors, "-", 2.5),
        ("I2_hat (decision disagreement)", vote_dis, errors, "--", 1.5),
        ("Max confidence (T_hat)", 1 - P1, errors, "-.", 1.5),
        ("Margin", 1 - (P1 - P2), errors, ":", 1.5),
        ("LR alone", 1 - np.max(P_lr, axis=1), errs_lr, "-", 2.5),
    ]:
        cov, risk = risk_coverage_tiecorrected(score, errs)
        ax.plot(cov, risk, style, label=name, lw=lw)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Risk (expected error rate among accepted)")
    ax.set_title("Selective risk-coverage curves -- JNU (held-out 1000 rpm)", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig5_RiskCoverage_JNU_v7.png"), dpi=300, bbox_inches="tight")
    print("Saved Fig5_RiskCoverage_JNU_v7.png")
