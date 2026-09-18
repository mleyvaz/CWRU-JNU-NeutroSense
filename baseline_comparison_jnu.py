# -*- coding: utf-8 -*-
"""
Baseline comparison (added 2026-09-17, external review point 5): compares
the proposed indicators against simple, conventional single-score selectors
on JNU (the only benchmark where selective prediction is non-trivial, since
CWRU now classifies at ~100% accuracy after the file-mapping fix). For each
candidate score, instances are accepted in order of decreasing confidence
(lowest uncertainty first) and we report the area under the risk-coverage
curve (AURC, using error rate as risk; lower is better) computed over the
full coverage range [0,1], a standard selective-classification summary
(Geifman & El-Yaniv, ref [10]), plus accuracy at 50% coverage as a fixed
reference point. Reuses the exact same JNU leave-1000rpm-out split, scaler-
then-SMOTE order, and RF+XGB+LR ensemble as pipeline_v3_grouped.py.
"""
import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

import jnu_neutro_pipeline as jn
from pipeline_v3_grouped import build_jnu_grouped

OUT = os.path.dirname(os.path.abspath(__file__))


def aurc(scores_ascending_is_confident, errors):
    """scores_ascending_is_confident: lower value = more confident (accept first).
    Returns (AURC, acc_at_50pct_coverage).

    NOTE (fixed 2026-09-17, external review): a plain np.argsort breaks ties
    between equal scores using array order, which is an arbitrary and
    unstated tie-break -- consequential here because I2_hat only takes 4
    discrete values, so it has large tie groups (e.g. all 1,655 unanimous-
    agreement JNU instances share one score). We instead compute the risk
    within each tie group as its EXPECTED value under uniform-random
    tie-breaking: for a group of m tied instances with e errors entered
    after (before_n, before_err) instances have already been accepted, the
    expected cumulative error after accepting j of the m (drawn without
    replacement) is before_err + j*e/m, matching the closed-form expectation
    over random permutations rather than any single arbitrary order."""
    order = np.argsort(scores_ascending_is_confident, kind="stable")
    scores_sorted = np.asarray(scores_ascending_is_confident)[order]
    errors_sorted = errors[order]
    n = len(errors_sorted)

    cum_err_expected = np.empty(n, dtype=float)
    before_n, before_err = 0, 0.0
    i = 0
    while i < n:
        j = i
        while j < n and scores_sorted[j] == scores_sorted[i]:
            j += 1
        m = j - i
        e = errors_sorted[i:j].sum()
        for k in range(1, m + 1):
            cum_err_expected[i + k - 1] = before_err + k * e / m
        before_n += m
        before_err += e
        i = j

    coverages = np.arange(1, n + 1) / n
    risks = cum_err_expected / np.arange(1, n + 1)
    area = np.trapezoid(risks, coverages) if hasattr(np, "trapezoid") else np.trapz(risks, coverages)
    idx50 = int(n * 0.5) - 1
    acc50 = 1 - risks[idx50]
    return area, acc50


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

    P_rf = rf.predict_proba(Xte_sc)
    P_xgb = xgb.predict_proba(Xte_sc)
    P_lr = lr.predict_proba(Xte_sc)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != yte).astype(int)
    print(f"Ensemble accuracy check: {accuracy_score(yte, y_pred)*100:.2f}% (should be 40.64%)")

    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    T_hat = P1
    margin = P1 - P2
    K = P_avg.shape[1]
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(K)

    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1),
                       np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_disagreement = 1.0 - agree / 3.0

    combined = entropy + vote_disagreement  # simple unweighted combination

    print(f"\n{'Selector':30s} {'AURC (lower=better)':22s} {'Acc@50% coverage':18s}")
    for name, score in [
        ("Max confidence (1-T_hat)", 1 - T_hat),
        ("Margin (1-margin)", 1 - margin),
        ("Predictive entropy (I1_hat)", entropy),
        ("Vote disagreement (I2_hat)", vote_disagreement),
        ("Combined (I1_hat + I2_hat)", combined),
        ("Logistic Regression alone (1-P_lr max)", 1 - np.max(P_lr, axis=1)),
    ]:
        if name.startswith("Logistic"):
            errs_lr = (np.argmax(P_lr, axis=1) != yte).astype(int)
            area, acc50 = aurc(score, errs_lr)
        else:
            area, acc50 = aurc(score, errors)
        print(f"{name:30s} {area:.4f}                {acc50*100:.2f}%")
