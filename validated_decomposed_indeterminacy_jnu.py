# -*- coding: utf-8 -*-
"""
Critical follow-up to decomposed_indeterminacy_jnu.py. That script found a
much larger oracle joint-rule improvement (AURC 0.042297) using I1_new =
mean_m[H(P_m)] and I2_new = mean_m[KL(P_m||P_avg)] instead of the original
(total entropy, discrete vote-disagreement) pair -- far bigger than any
previous oracle result in this project (0.000046 with 3 models, 0.001490
with 8 models). Every previous oracle-only positive result in this project
failed to survive honest validation (Sections 4.6.1, 4.6.3b). This script
runs the SAME check on the new indicators: fit on 600rpm, select
thresholds on 800rpm (validation), evaluate the FIXED rule on 1000rpm
(test, untouched).
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
from baseline_comparison_jnu import aurc

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


def entropy_norm(P, n_classes):
    return -np.sum(P * np.log(P + 1e-12), axis=1) / np.log(n_classes)


def kl_div(P, Q):
    P = np.clip(P, 1e-12, 1.0)
    Q = np.clip(Q, 1e-12, 1.0)
    return np.sum(P * np.log(P / Q), axis=1)


def compute_new_indicators(rf, xgb, lr, X):
    P_rf = rf.predict_proba(X); P_xgb = xgb.predict_proba(X); P_lr = lr.predict_proba(X)
    P_avg = (P_rf + P_xgb + P_lr) / 3.0
    y_pred = np.argmax(P_avg, axis=1)
    n_classes = P_avg.shape[1]
    H_rf = entropy_norm(P_rf, n_classes); H_xgb = entropy_norm(P_xgb, n_classes); H_lr = entropy_norm(P_lr, n_classes)
    I1_new = (H_rf + H_xgb + H_lr) / 3.0
    kl_rf = kl_div(P_rf, P_avg) / np.log(n_classes)
    kl_xgb = kl_div(P_xgb, P_avg) / np.log(n_classes)
    kl_lr = kl_div(P_lr, P_avg) / np.log(n_classes)
    I2_new = (kl_rf + kl_xgb + kl_lr) / 3.0
    return y_pred, I1_new, I2_new


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

    y_pred_val, I1_val, I2_val = compute_new_indicators(rf, xgb, lr, X_val_sc)
    errors_val = (y_pred_val != y_val).astype(int)
    print(f"Ensemble accuracy on validation (800 rpm): {(1-errors_val.mean())*100:.2f}%")

    y_pred_test, I1_test, I2_test = compute_new_indicators(rf, xgb, lr, X_test_sc)
    errors_test = (y_pred_test != y_test).astype(int)
    print(f"Ensemble accuracy on test (1000 rpm, never used to fit or select thresholds): "
          f"{(1-errors_test.mean())*100:.2f}%")

    # NEW (addresses the most serious independent-review finding on this experiment):
    # the earlier version of this script validated single-threshold POINTS for I1_new-only,
    # I2_new-only, and the joint rule, but never validated the actual AURC of the fixed
    # linear COMBINATION reported as this project's headline positive result
    # (decomposed_indeterminacy_jnu.py). We do that here: compute I1_new/I2_new's own
    # mean/std on the FIT set (600rpm, the only data the ensemble and any deployed
    # standardization constants would ever see), use those FIXED constants to standardize
    # validation and test, and report full AURC (not just a single coverage point) on both.
    y_pred_fit, I1_fit, I2_fit = compute_new_indicators(rf, xgb, lr, X_fit_sc)
    combo_val = (I1_val - I1_fit.mean()) / I1_fit.std() + (I2_val - I2_fit.mean()) / I2_fit.std()
    combo_test = (I1_test - I1_fit.mean()) / I1_fit.std() + (I2_test - I2_fit.mean()) / I2_fit.std()
    area_combo_val, acc50_combo_val = aurc(combo_val, errors_val)
    area_combo_test, acc50_combo_test = aurc(combo_test, errors_test)
    area_i1_test_full, _ = aurc(I1_test, errors_test)
    area_i2_test_full, _ = aurc(I2_test, errors_test)
    print(f"\n=== AURC of the FIXED (fit-standardized) combination, full risk-coverage curve ===")
    print(f"  Validation (sanity check, 800rpm): AURC={area_combo_val:.4f}  Acc@50%cov={acc50_combo_val*100:.2f}%")
    print(f"  TEST (honest, 1000rpm, never used to fit standardization constants): "
          f"AURC={area_combo_test:.4f}  Acc@50%cov={acc50_combo_test*100:.2f}%")
    print(f"  For comparison, same fit-600/test-1000 ensemble: I1_new alone AURC={area_i1_test_full:.4f}, "
          f"I2_new alone AURC={area_i2_test_full:.4f}")
    if area_combo_test < min(area_i1_test_full, area_i2_test_full) - 1e-9:
        print(f"  => The fixed combination's AURC genuinely improves on both single indicators here too, "
              f"using ONLY fit-set statistics -- this is the properly validated version of Section 4.6.1's "
              f"headline result.")
    else:
        print(f"  => The fixed combination does NOT clearly improve on the better single indicator under "
              f"this fit-only-standardized, single-condition-ensemble check.")

    # --- Rule 1: I1_new-only threshold, selected on validation for 50% coverage ---
    tau1_val_50 = np.quantile(I1_val, 0.50)
    test_mask_i1 = I1_test <= tau1_val_50
    test_coverage_i1 = test_mask_i1.mean()
    test_risk_i1 = errors_test[test_mask_i1].mean() if test_mask_i1.sum() > 0 else float("nan")
    val_risk_i1 = errors_val[I1_val <= tau1_val_50].mean()
    print(f"\nI1_NEW-ONLY rule: tau1 = 50th percentile of VALIDATION I1_new = {tau1_val_50:.6f}")
    print(f"  Validation (sanity check): coverage={(I1_val<=tau1_val_50).mean()*100:.2f}%  risk={val_risk_i1*100:.2f}%")
    print(f"  TEST (honest): coverage={test_coverage_i1*100:.2f}%  risk={test_risk_i1*100:.2f}%  "
          f"(N accepted={int(test_mask_i1.sum())} of {len(y_test)})")

    # --- Rule 1b: I2_new-only threshold, selected on validation for 50% coverage ---
    tau2_only_val_50 = np.quantile(I2_val, 0.50)
    test_mask_i2only = I2_test <= tau2_only_val_50
    test_coverage_i2only = test_mask_i2only.mean()
    test_risk_i2only = errors_test[test_mask_i2only].mean() if test_mask_i2only.sum() > 0 else float("nan")
    print(f"\nI2_NEW-ONLY rule: tau2 = 50th percentile of VALIDATION I2_new = {tau2_only_val_50:.6f}")
    print(f"  TEST (honest): coverage={test_coverage_i2only*100:.2f}%  risk={test_risk_i2only*100:.2f}%")

    # --- Rule 2: joint (tau1, tau2) AND-rule, grid-selected on validation only, exact grid ---
    tau1_grid = np.unique(I1_val)
    tau2_grid = np.quantile(I2_val, np.linspace(0.0, 1.0, 300))  # I2_new is continuous; use a fine but bounded grid
    tau2_grid = np.unique(tau2_grid)
    best = None
    for t2 in tau2_grid:
        mask_base = I2_val <= t2
        if mask_base.mean() < 0.50:
            continue
        # for this t2, find best t1 among unique I1_val restricted to mask_base
        for t1 in tau1_grid:
            mask = mask_base & (I1_val <= t1)
            cov = mask.mean()
            if cov < 0.50:
                continue
            risk = errors_val[mask].mean()
            if best is None or risk < best[0]:
                best = (risk, t1, t2, cov)
    if best is None:
        best = (val_risk_i1, tau1_val_50, np.quantile(I2_val, 1.0), (I1_val <= tau1_val_50).mean())
    val_risk_joint, tau1_joint, tau2_joint, val_cov_joint = best
    print(f"\nJOINT rule (NEW indicators): (tau1, tau2) selected on VALIDATION to minimize risk at >=50% coverage: "
          f"tau1={tau1_joint:.6f}, tau2={tau2_joint:.6f}")
    print(f"  Validation (sanity check): coverage={val_cov_joint*100:.2f}%  risk={val_risk_joint*100:.2f}%")
    test_mask_joint = (I1_test <= tau1_joint) & (I2_test <= tau2_joint)
    test_coverage_joint = test_mask_joint.mean()
    test_risk_joint = errors_test[test_mask_joint].mean() if test_mask_joint.sum() > 0 else float("nan")
    print(f"  TEST (honest): coverage={test_coverage_joint*100:.2f}%  risk={test_risk_joint*100:.2f}%  "
          f"(N accepted={int(test_mask_joint.sum())} of {len(y_test)})")

    print("\n=== SUMMARY: does the NEW-indicator joint rule survive honest validation? ===")
    print(f"  I1_new-ONLY rule: test coverage={test_coverage_i1*100:.2f}%  test risk={test_risk_i1*100:.2f}%")
    print(f"  I2_new-ONLY rule: test coverage={test_coverage_i2only*100:.2f}%  test risk={test_risk_i2only*100:.2f}%")
    print(f"  JOINT rule:       test coverage={test_coverage_joint*100:.2f}%  test risk={test_risk_joint*100:.2f}%")
    best_single_risk = min(test_risk_i1, test_risk_i2only)
    if test_risk_joint < best_single_risk - 1e-9:
        verdict = (f"YES -- the JOINT rule with the NEW (decomposed) indicators achieves LOWER test risk "
                   f"({test_risk_joint*100:.2f}%) than either single indicator alone "
                   f"({best_single_risk*100:.2f}%): a genuinely validated improvement.")
    elif abs(test_risk_joint - best_single_risk) <= 1e-9:
        verdict = "TIE -- no validated improvement over the better single indicator."
    else:
        verdict = (f"NO -- even with the new decomposed indicators, the JOINT rule does not beat the "
                   f"better single indicator under honest validation ({test_risk_joint*100:.2f}% vs "
                   f"{best_single_risk*100:.2f}%).")
    print(f"  => {verdict}")
