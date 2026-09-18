# -*- coding: utf-8 -*-
"""
Tests whether a genuinely JOINT decision rule over (I1_hat, I2_hat) -- accept
only if I1_hat <= tau1 AND I2_hat <= tau2 -- outperforms any single-score
selector (Section 4.4's baselines: T_hat, margin, I1_hat alone, I2_hat
alone, and the linear combination I1_hat+I2_hat). This directly tests
whether treating aleatoric and epistemic uncertainty as two SEPARATE axes
(rather than collapsing them into one score) has a demonstrable selective-
classification advantage -- the specific, previously untested claim needed
to argue for a genuinely multi-component ("neutrosophic") decision rule
rather than a single scalar uncertainty score under a different name.

Method: sweep a grid of (tau1, tau2) pairs, compute the resulting (coverage,
risk) for the joint AND-rule at each pair, take the lower (risk-minimizing)
envelope of all pairs at each coverage level to get the JOINT rule's own
risk-coverage frontier, and compare its AURC against the single-score
curves computed the same way (same JNU leave-1000rpm-out split, ensemble).
"""
import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

from pipeline_v3_grouped import build_jnu_grouped
from baseline_comparison_jnu import aurc

OUT = os.path.dirname(os.path.abspath(__file__))

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
    n = len(errors)

    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0

    # Joint AND-rule: sweep (tau1, tau2), record (coverage, risk) for each pair.
    tau1_grid = np.quantile(entropy, np.linspace(0.01, 1.0, 100))
    tau2_grid = np.unique(vote_dis)  # only 4 discrete values: 0, 1/3, 2/3, 1
    points = []
    for t2 in tau2_grid:
        for t1 in tau1_grid:
            mask = (entropy <= t1) & (vote_dis <= t2)
            c = mask.mean()
            if c == 0:
                continue
            r = errors[mask].mean()
            points.append((c, r))
    points = np.array(points)

    # Lower envelope: for each coverage level, the minimum achievable risk
    # across all (tau1, tau2) pairs reaching at least that coverage.
    order = np.argsort(points[:, 0])
    points = points[order]
    cov_grid = np.linspace(0.01, 1.0, 200)
    env_risk = []
    for c in cov_grid:
        feasible = points[points[:, 0] >= c]
        if len(feasible) == 0:
            feasible = points[-1:]
        env_risk.append(feasible[:, 1].min())
    env_risk = np.array(env_risk)
    aurc_joint = np.trapezoid(env_risk, cov_grid) if hasattr(np, "trapezoid") else np.trapz(env_risk, cov_grid)
    idx50 = np.argmin(np.abs(cov_grid - 0.5))
    acc50_joint = 1 - env_risk[idx50]

    print(f"\nJoint AND-rule (I1_hat<=tau1 AND I2_hat<=tau2) lower-envelope frontier:")
    print(f"  AURC = {aurc_joint:.4f}   Acc@50% coverage = {acc50_joint*100:.2f}%")

    print(f"\nFor comparison (single-score selectors, same tie-corrected AURC method):")
    for name, score in [
        ("I1_hat alone", entropy),
        ("I2_hat alone", vote_dis),
        ("Max confidence (T_hat)", 1 - P1),
        ("Margin", 1 - (P1 - P2)),
    ]:
        area, acc50 = aurc(score, errors)
        print(f"  {name:28s} AURC={area:.4f}  Acc@50%cov={acc50*100:.2f}%")

    improvement = None
    # Reference: I1_hat alone AURC from baseline_comparison_jnu.py = 0.3915
    ref_area, _ = aurc(entropy, errors)
    if aurc_joint < ref_area:
        print(f"\n=> Joint rule IMPROVES on I1_hat alone: {aurc_joint:.4f} < {ref_area:.4f}")
    else:
        print(f"\n=> Joint rule does NOT improve on I1_hat alone: {aurc_joint:.4f} >= {ref_area:.4f}")
