# -*- coding: utf-8 -*-
"""
Extension requested after the 10-round adversarial review closed: every
selective-classification threshold evaluated elsewhere in this project
(Section 4.4's baseline comparison, the oracle joint-rule test) is chosen
BY SEARCHING THE TEST SET ITSELF -- an honest upper bound, but not a
genuinely validated decision rule. This script builds and evaluates a rule
the conventional way: fit thresholds on a validation condition the rule
never sees again, then apply the FIXED rule, unmodified, to a third,
still-independent test condition.

Three-way condition split (JNU): 600 rpm (fit ensemble) -> 800 rpm (select
thresholds) -> 1000 rpm (final, untouched evaluation). This differs from
the rest of the paper's 600+800-train / 1000-test protocol -- it uses only
600 rpm to fit the ensemble, so its raw accuracy numbers are not directly
comparable to Table 1 and are not meant to replace them; this script tests
whether a rule selected on 800 rpm alone generalizes to 1000 rpm, which is
what "validated" is supposed to mean.

Two rules are compared, both selected on validation only:
  I1-ONLY  : single threshold tau1 on predictive entropy, chosen on
             validation to be the coverage-50% point (nearest achievable).
  JOINT    : grid-search (tau1, tau2) on validation ONLY, selecting the
             pair whose validation risk is lowest among those reaching at
             least 50% validation coverage.
Both fixed rules are then applied, unchanged, to the 1000 rpm test set, and
their resulting coverage/risk are reported. This is the first genuinely
validated (not oracle/retrospective) test of whether the joint I1_hat/I2_hat
rule beats the single-score rule.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

import jnu_neutro_pipeline as jn

OUT = os.path.dirname(os.path.abspath(__file__))

FIT_FILES = {"n600_3_2.csv", "ib600_2.csv", "ob600_2.csv", "tb600_2.csv"}
VAL_FILES = {"n800_3_2.csv", "ib800_2.csv", "ob800_2.csv", "tb800_2.csv"}
TEST_FILES = {"n1000_3_2.csv", "ib1000_2.csv", "ob1000_2.csv", "tb1000_2.csv"}


def build_three_way():
    jn.download_files()
    sets = {"fit": ([], []), "val": ([], []), "test": ([], [])}
    for fname, label, lname in jn.JNU_FILES:
        fpath = os.path.join(jn.DATA_DIR, fname)
        sig = pd.read_csv(fpath, header=None).values.flatten()
        n = (len(sig) - jn.WINDOW) // jn.STEP + 1
        if fname in FIT_FILES:
            key = "fit"
        elif fname in VAL_FILES:
            key = "val"
        elif fname in TEST_FILES:
            key = "test"
        else:
            continue
        rows, ys = sets[key]
        for i in range(n):
            rows.append(jn.extract_features(sig[i * jn.STEP: i * jn.STEP + jn.WINDOW]))
            ys.append(label)
        print(f"  {fname}: {n} windows, class={lname} -> {key}")
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in sets.items()}


def ensemble_scores(rf, xgb, lr, X):
    P_rf = rf.predict_proba(X); P_xgb = xgb.predict_proba(X); P_lr = lr.predict_proba(X)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0
    return y_pred, entropy, vote_dis


if __name__ == "__main__":
    print("Building three-way JNU split (600 rpm fit / 800 rpm validation / 1000 rpm test)...")
    data = build_three_way()
    X_fit, y_fit = data["fit"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]
    print(f"\nFit: N={len(y_fit)}  Validation: N={len(y_val)}  Test: N={len(y_test)}")

    scaler = StandardScaler()
    X_fit_sc = scaler.fit_transform(X_fit)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    smote = SMOTE(random_state=42)
    Xs_sc, ys = smote.fit_resample(X_fit_sc, y_fit)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    rf.fit(Xs_sc, ys); xgb.fit(Xs_sc, ys); lr.fit(Xs_sc, ys)

    y_pred_fit, _, _ = ensemble_scores(rf, xgb, lr, X_fit_sc)
    print(f"Ensemble accuracy on its own fit condition (600 rpm, in-sample, sanity check): "
          f"{(y_pred_fit == y_fit).mean()*100:.2f}%")

    y_pred_val, entropy_val, vote_dis_val = ensemble_scores(rf, xgb, lr, X_val_sc)
    errors_val = (y_pred_val != y_val).astype(int)
    print(f"Ensemble accuracy on validation (800 rpm, never used to fit): {(1-errors_val.mean())*100:.2f}%")

    y_pred_test, entropy_test, vote_dis_test = ensemble_scores(rf, xgb, lr, X_test_sc)
    errors_test = (y_pred_test != y_test).astype(int)
    print(f"Ensemble accuracy on test (1000 rpm, never used to fit OR select thresholds): "
          f"{(1-errors_test.mean())*100:.2f}%")

    # --- Rule 1: I1-only threshold, selected on validation to hit ~50% coverage ---
    tau1_val_50 = np.quantile(entropy_val, 0.50)
    print(f"\nI1-ONLY rule: tau1 = 50th percentile of VALIDATION entropy = {tau1_val_50:.6f}")
    val_coverage_i1 = (entropy_val <= tau1_val_50).mean()
    val_risk_i1 = errors_val[entropy_val <= tau1_val_50].mean() if val_coverage_i1 > 0 else float("nan")
    print(f"  On validation (sanity check): coverage={val_coverage_i1*100:.2f}%  risk={val_risk_i1*100:.2f}%")
    test_mask_i1 = entropy_test <= tau1_val_50
    test_coverage_i1 = test_mask_i1.mean()
    test_risk_i1 = errors_test[test_mask_i1].mean() if test_mask_i1.sum() > 0 else float("nan")
    print(f"  On TEST (honest, threshold fixed before looking at test): coverage={test_coverage_i1*100:.2f}%  "
          f"risk={test_risk_i1*100:.2f}%  (N accepted={int(test_mask_i1.sum())} of {len(y_test)})")

    # --- Rule 2: joint (tau1, tau2) AND-rule, grid-selected on validation only ---
    # IMPORTANT: tau1 candidates must be the FULL set of unique validation entropy
    # values (exact), not a coarse quantile subsample -- a coarser tau1 grid here
    # than the exact value used for the I1-ONLY rule would repeat exactly the kind
    # of grid-resolution artifact an external review caught and required fixing in
    # joint_decision_rule_jnu.py. With an exact grid, tau2=max must recover at
    # least the I1-ONLY rule's own validation risk (the AND-rule family contains
    # the univariate family as the tau2=max special case).
    tau1_grid = np.unique(entropy_val)
    tau2_grid = np.unique(vote_dis_val)
    best = None  # (val_risk, tau1, tau2, val_coverage)
    for t2 in tau2_grid:
        for t1 in tau1_grid:
            mask = (entropy_val <= t1) & (vote_dis_val <= t2)
            cov = mask.mean()
            if cov < 0.50:
                continue
            risk = errors_val[mask].mean()
            if best is None or risk < best[0]:
                best = (risk, t1, t2, cov)
    if best is None:
        # fallback: no combination reaches 50% coverage on validation; take the
        # highest-coverage combination available instead of leaving the rule undefined.
        for t2 in tau2_grid:
            for t1 in tau1_grid:
                mask = (entropy_val <= t1) & (vote_dis_val <= t2)
                cov = mask.mean()
                risk = errors_val[mask].mean() if cov > 0 else float("inf")
                if best is None or cov > best[3]:
                    best = (risk, t1, t2, cov)
    val_risk_joint, tau1_joint, tau2_joint, val_cov_joint = best
    print(f"\nJOINT rule: (tau1, tau2) selected on VALIDATION to minimize risk at >=50% coverage: "
          f"tau1={tau1_joint:.6f}, tau2={tau2_joint:.4f}")
    print(f"  On validation (sanity check): coverage={val_cov_joint*100:.2f}%  risk={val_risk_joint*100:.2f}%")
    test_mask_joint = (entropy_test <= tau1_joint) & (vote_dis_test <= tau2_joint)
    test_coverage_joint = test_mask_joint.mean()
    test_risk_joint = errors_test[test_mask_joint].mean() if test_mask_joint.sum() > 0 else float("nan")
    print(f"  On TEST (honest, thresholds fixed before looking at test): coverage={test_coverage_joint*100:.2f}%  "
          f"risk={test_risk_joint*100:.2f}%  (N accepted={int(test_mask_joint.sum())} of {len(y_test)})")

    print("\n=== SUMMARY: validated (not oracle) rule comparison on TRUE held-out test ===")
    print(f"  I1-ONLY rule:  test coverage={test_coverage_i1*100:.2f}%  test risk={test_risk_i1*100:.2f}%")
    print(f"  JOINT rule:    test coverage={test_coverage_joint*100:.2f}%  test risk={test_risk_joint*100:.2f}%")
    if test_risk_joint < test_risk_i1 - 1e-9:
        verdict = "JOINT rule achieves LOWER test risk than I1-only (genuine validated improvement)"
    elif abs(test_risk_joint - test_risk_i1) <= 1e-9:
        verdict = "JOINT rule ties I1-only on test risk (no validated improvement)"
    else:
        verdict = "JOINT rule achieves HIGHER test risk than I1-only (no validated improvement -- worse, in fact)"
    print(f"  => {verdict}")
    print("\nNote: this experiment uses a DIFFERENT protocol (600rpm-fit/800rpm-validation/1000rpm-test) than "
          "the main paper (600+800rpm-train/1000rpm-test), so its raw accuracy numbers are not directly "
          "comparable to Table 1 -- it is a validation-methodology check, not a replacement main result.")
