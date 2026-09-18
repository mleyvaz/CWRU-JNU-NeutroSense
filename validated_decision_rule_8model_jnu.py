# -*- coding: utf-8 -*-
"""
Direct follow-up question after Section 4.6: does the 8-model ensemble's
statistically significant I2_hat signal (larger_ensemble_i2_jnu.py: partial
r=+0.0712, p=4.85e-08, oracle joint-rule AURC improvement of 0.001490)
survive a genuinely VALIDATED protocol, or is it still only an oracle
(test-set-optimized) artifact like the 3-model result in
validated_decision_rule_jnu.py?

Same three-way JNU split as validated_decision_rule_jnu.py (fit on 600rpm,
select thresholds on 800rpm, evaluate the FIXED rule on 1000rpm), but using
the 8-model ensemble (RF, XGBoost, LR, SVM, k-NN, Gaussian NB, Extra Trees,
Gradient Boosting) from larger_ensemble_i2_jnu.py instead of the 3-model
one. This is the correct test of whether more base learners turn the
oracle-only positive result in Section 4.6.3 into a genuinely validated
one.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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


def ensemble_scores(models, X):
    probas = [m.predict_proba(X) for m in models.values()]
    P_avg = np.mean(probas, axis=0)
    y_pred = np.argmax(P_avg, axis=1)
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(p, axis=1) for p in probas], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / len(models)
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

    models = {
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="mlogloss", verbosity=0),
        "LR": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "NB": GaussianNB(),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=42),
    }
    print(f"\nFitting {len(models)}-model ensemble on 600rpm only...")
    for name, m in models.items():
        m.fit(Xs_sc, ys)

    y_pred_val, entropy_val, vote_dis_val = ensemble_scores(models, X_val_sc)
    errors_val = (y_pred_val != y_val).astype(int)
    print(f"8-model ensemble accuracy on validation (800 rpm, never used to fit): {(1-errors_val.mean())*100:.2f}%")

    y_pred_test, entropy_test, vote_dis_test = ensemble_scores(models, X_test_sc)
    errors_test = (y_pred_test != y_test).astype(int)
    print(f"8-model ensemble accuracy on test (1000 rpm, never used to fit OR select thresholds): "
          f"{(1-errors_test.mean())*100:.2f}%")
    print(f"I2_hat discrete levels with 8 models: {len(np.unique(vote_dis_val))}")

    # --- Rule 1: I1-only threshold, selected on validation for ~50% coverage ---
    tau1_val_50 = np.quantile(entropy_val, 0.50)
    test_mask_i1 = entropy_test <= tau1_val_50
    test_coverage_i1 = test_mask_i1.mean()
    test_risk_i1 = errors_test[test_mask_i1].mean() if test_mask_i1.sum() > 0 else float("nan")
    val_mask_i1 = entropy_val <= tau1_val_50
    val_risk_i1 = errors_val[val_mask_i1].mean()
    print(f"\nI1-ONLY rule: tau1 = 50th percentile of VALIDATION entropy = {tau1_val_50:.6f}")
    print(f"  Validation (sanity check): coverage={val_mask_i1.mean()*100:.2f}%  risk={val_risk_i1*100:.2f}%")
    print(f"  TEST (honest): coverage={test_coverage_i1*100:.2f}%  risk={test_risk_i1*100:.2f}%  "
          f"(N accepted={int(test_mask_i1.sum())} of {len(y_test)})")

    # --- Rule 2: joint (tau1, tau2) AND-rule, grid-selected on validation only, exact grid ---
    tau1_grid = np.unique(entropy_val)
    tau2_grid = np.unique(vote_dis_val)
    best = None
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
    print(f"  Validation (sanity check): coverage={val_cov_joint*100:.2f}%  risk={val_risk_joint*100:.2f}%")
    test_mask_joint = (entropy_test <= tau1_joint) & (vote_dis_test <= tau2_joint)
    test_coverage_joint = test_mask_joint.mean()
    test_risk_joint = errors_test[test_mask_joint].mean() if test_mask_joint.sum() > 0 else float("nan")
    print(f"  TEST (honest): coverage={test_coverage_joint*100:.2f}%  risk={test_risk_joint*100:.2f}%  "
          f"(N accepted={int(test_mask_joint.sum())} of {len(y_test)})")

    print("\n=== SUMMARY: does the 8-model ensemble's I2_hat signal survive honest validation? ===")
    print(f"  I1-ONLY rule:  test coverage={test_coverage_i1*100:.2f}%  test risk={test_risk_i1*100:.2f}%")
    print(f"  JOINT rule:    test coverage={test_coverage_joint*100:.2f}%  test risk={test_risk_joint*100:.2f}%")
    if test_risk_joint < test_risk_i1 - 1e-9:
        verdict = (f"YES -- with 8 models, the JOINT rule achieves LOWER test risk than I1-only "
                   f"({test_risk_joint*100:.2f}% < {test_risk_i1*100:.2f}%): a genuinely validated "
                   f"improvement, not just an oracle one.")
    elif abs(test_risk_joint - test_risk_i1) <= 1e-9:
        verdict = "TIE -- no validated improvement either way."
    else:
        verdict = (f"NO -- even with 8 models, the JOINT rule does NOT beat I1-only under honest "
                   f"validation ({test_risk_joint*100:.2f}% vs {test_risk_i1*100:.2f}% risk); the "
                   f"oracle-only improvement reported in Section 4.6.3 does not survive validation.")
    print(f"  => {verdict}")
