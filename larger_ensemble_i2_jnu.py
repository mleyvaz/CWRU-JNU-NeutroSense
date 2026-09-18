# -*- coding: utf-8 -*-
"""
Extension requested after the 10-round adversarial review closed. I2_hat
(decision disagreement) with only 3 base learners takes just 4 discrete
values (0, 1/3, 2/3, 1) -- a coarse indicator, explicitly flagged as a
limitation in the manuscript's Limitations paragraph ("with only three
base learners, I2-hat is a coarse indicator; ensembles with more members
might reveal a more consistent disagreement signal"). This script tests
that directly: build an 8-model ensemble (RF, XGBoost, Logistic Regression,
SVM, k-NN, Gaussian Naive Bayes, Extra Trees, Gradient Boosting), giving
I2_hat 9 discrete levels (0/8 ... 8/8 disagreement) instead of 4, and
re-run the same partial-correlation and AURC baseline comparison used in
Sections 4.3-4.4 to see whether finer granularity improves I2_hat's
independent contribution or its selective-classification performance.
"""
import os
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
np.random.seed(42)

from pipeline_v3_grouped import build_jnu_grouped, multivariate_partial_r
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
    n_models = len(models)
    print(f"\nFitting {n_models}-model ensemble: {list(models.keys())}")
    probas = {}
    preds = {}
    for name, m in models.items():
        m.fit(Xs_sc, ys)
        P = m.predict_proba(Xte_sc)
        probas[name] = P
        preds[name] = np.argmax(P, axis=1)
        acc = (preds[name] == yte).mean()
        print(f"  {name}: test accuracy = {acc*100:.2f}%")

    P_avg = np.mean(list(probas.values()), axis=0)
    y_pred = np.argmax(P_avg, axis=1)
    errors = (y_pred != yte).astype(int)
    print(f"\n{n_models}-model ensemble (soft-vote average) accuracy: {(1-errors.mean())*100:.2f}%")

    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([preds[name] for name in models], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / n_models
    print(f"I2_hat (decision disagreement) with {n_models} models: {len(np.unique(vote_dis))} discrete levels "
          f"(was 4 with 3 models) -- values: {sorted(np.unique(vote_dis).tolist())}")

    sorted_idx = np.argsort(-P_avg, axis=1)
    P1 = np.take_along_axis(P_avg, sorted_idx[:, [0]], axis=1).ravel()
    P2 = np.take_along_axis(P_avg, sorted_idx[:, [1]], axis=1).ravel()
    T_hat, F_hat = P1, P2

    # Partial correlation of I2_hat with error, controlling for T_hat, F_hat, I1_hat --
    # same protocol as Section 4.5, just with the finer-grained I2_hat.
    partial_r, partial_p = multivariate_partial_r(vote_dis, [T_hat, F_hat, entropy], errors)
    raw_r, raw_p = pearsonr(vote_dis, errors)
    print(f"\nI2_hat (8-model) raw r with error = {raw_r:.4f} (p={raw_p:.3g})")
    print(f"I2_hat (8-model) partial r | T,F,I1 = {partial_r:.4f} (p={partial_p:.3g})")

    print(f"\nFor comparison, with the original 3-model ensemble (from baseline_comparison_output_r2.log): "
          f"I2_hat AURC=0.4452, partial r=+0.017 (p=0.20, n.s.)")

    area_i2, acc50_i2 = aurc(vote_dis, errors)
    area_i1, acc50_i1 = aurc(entropy, errors)
    print(f"\n8-model I2_hat alone:  AURC={area_i2:.4f}  Acc@50%cov={acc50_i2*100:.2f}%")
    print(f"8-model I1_hat alone:  AURC={area_i1:.4f}  Acc@50%cov={acc50_i1*100:.2f}%")

    if area_i2 < 0.4452 - 1e-9:
        verdict = f"IMPROVES over the 3-model I2_hat's AURC (0.4452 -> {area_i2:.4f})"
    else:
        verdict = f"does NOT improve over the 3-model I2_hat's AURC (0.4452 -> {area_i2:.4f})"
    print(f"\n=> Finer-grained (8-model, 9-level) I2_hat {verdict}.")

    # Now that I2_hat has a significant independent contribution with 8 models,
    # re-test whether a joint I1_hat/I2_hat rule beats I1_hat alone in this MORE
    # favorable regime -- using the same exact, tie-corrected method as
    # joint_decision_rule_jnu.py (not a coarse grid) to avoid repeating that bug.
    from joint_decision_rule_jnu import tie_corrected_local_risk
    n_total = len(errors)
    tau2_grid = np.unique(vote_dis)
    branch_risk_by_k = {}
    for t2 in tau2_grid:
        mask_t2 = vote_dis <= t2
        branch_risk_by_k[t2] = tie_corrected_local_risk(entropy[mask_t2], errors[mask_t2])
    env_risk = np.full(n_total, np.inf)
    for t2, local_risk in branch_risk_by_k.items():
        env_risk[:len(local_risk)] = np.minimum(env_risk[:len(local_risk)], local_risk)
    coverage = np.arange(1, n_total + 1) / n_total
    aurc_joint = np.trapezoid(env_risk, coverage) if hasattr(np, "trapezoid") else np.trapz(env_risk, coverage)
    print(f"\n8-model exact oracle joint rule (I1_hat<=tau1 AND I2_hat<=tau2): AURC={aurc_joint:.6f}")
    print(f"8-model I1_hat alone (reference, same method):                    AURC={area_i1:.6f}")
    diff = area_i1 - aurc_joint
    if diff > 1e-9:
        print(f"=> Joint rule IMPROVES on I1_hat alone by {diff:.6f} (8-model ensemble)")
    else:
        print(f"=> Joint rule does NOT meaningfully improve on I1_hat alone (diff={diff:.6f})")

    # Linear combination I1_hat + I2_hat (z-scored, unweighted sum), same as
    # Section 4.4's baseline comparison.
    combo = (entropy - entropy.mean()) / entropy.std() + (vote_dis - vote_dis.mean()) / vote_dis.std()
    area_combo, acc50_combo = aurc(combo, errors)
    print(f"\n8-model I1_hat + I2_hat (linear combination): AURC={area_combo:.6f}  Acc@50%cov={acc50_combo*100:.2f}%")
    if area_combo < area_i1 - 1e-9:
        print(f"=> Linear combination IMPROVES on I1_hat alone ({area_i1:.6f} -> {area_combo:.6f})")
    else:
        print(f"=> Linear combination does NOT improve on I1_hat alone ({area_i1:.6f} vs {area_combo:.6f})")
