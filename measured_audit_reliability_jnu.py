# -*- coding: utf-8 -*-
"""
Extension requested after the 10-round adversarial review closed. The
triage_cost_experiment.py analysis showed that any cost advantage of
routing by I2_hat depends entirely on an ASSUMED, illustrative audit
success probability (p_low, p_high) that was never measured from data.
This script replaces that assumption with a real, measured quantity.

Operationalization of "AUDIT" used here: consulting the standalone
Logistic Regression model's own prediction as a cheap "second opinion"
on instances where the full ensemble was wrong. This is a genuine,
zero-additional-training-cost audit mechanism (LR is already fit as part
of the ensemble), not a hypothetical external process. CAVEAT, stated
plainly: LR is also one of the three votes used to compute I2_hat itself,
so this is not a fully independent audit -- if I2_hat is high specifically
because LR disagreed with RF/XGB, and LR happens to be right, that will
mechanically show up as "audit succeeds more often when I2_hat is high",
which is a much weaker and more circular claim than an external check
would provide. We report this explicitly rather than hide it.

Measured quantity: among ensemble errors (JNU, leave-1000rpm-out test),
what fraction would have been corrected by using LR's own prediction
instead, separately for the lower-disagreement and higher-disagreement
groups defined the same way as in triage_cost_experiment.py? This gives
real p_low, p_high to replace the illustrative ones.
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
    y_pred_lr = np.argmax(P_lr, axis=1)
    errors = (y_pred != yte).astype(int)
    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0

    # Same flagged-set definition as triage_cost_experiment.py.
    tau1 = np.median(entropy)
    flagged = entropy > tau1
    n_flagged = int(flagged.sum())
    errors_flagged = errors[flagged]
    vote_dis_flagged = vote_dis[flagged]
    is_low_i2 = vote_dis_flagged <= np.median(vote_dis_flagged)

    # "Audit succeeds" on an ensemble ERROR if LR alone would have been correct.
    y_pred_lr_flagged = y_pred_lr[flagged]
    y_true_flagged = yte[flagged]
    ensemble_wrong = errors_flagged.astype(bool)
    lr_correct_when_ensemble_wrong = (y_pred_lr_flagged == y_true_flagged) & ensemble_wrong

    n_err_low = ensemble_wrong[is_low_i2].sum()
    n_err_high = ensemble_wrong[~is_low_i2].sum()
    n_audit_success_low = lr_correct_when_ensemble_wrong[is_low_i2].sum()
    n_audit_success_high = lr_correct_when_ensemble_wrong[~is_low_i2].sum()
    p_low_measured = n_audit_success_low / n_err_low if n_err_low > 0 else float("nan")
    p_high_measured = n_audit_success_high / n_err_high if n_err_high > 0 else float("nan")

    print(f"\nFlagged set (I1_hat > median): N={n_flagged}")
    print(f"Lower-disagreement group: N={int(is_low_i2.sum())}, ensemble errors={int(n_err_low)}, "
          f"of which LR-alone would have been correct: {int(n_audit_success_low)} "
          f"-> MEASURED p_low = {p_low_measured:.4f}")
    print(f"Higher-disagreement group: N={int((~is_low_i2).sum())}, ensemble errors={int(n_err_high)}, "
          f"of which LR-alone would have been correct: {int(n_audit_success_high)} "
          f"-> MEASURED p_high = {p_high_measured:.4f}")

    if not np.isnan(p_low_measured) and not np.isnan(p_high_measured):
        if p_high_measured < p_low_measured:
            direction = ("as assumed in the illustrative triage scenarios: audit (LR-alone) is LESS "
                         "reliable on the higher-disagreement group")
        elif p_high_measured > p_low_measured:
            direction = ("OPPOSITE of what the illustrative triage scenarios assumed: audit (LR-alone) is "
                         "MORE reliable on the higher-disagreement group, not less")
        else:
            direction = "no measured difference between groups"
        print(f"\nDirection: {direction}.")

    # Recompute the triage cost comparison using these MEASURED probabilities,
    # with the same cost structure as triage_cost_experiment.py's S1-S6, to see
    # whether the real, non-circular-caveat-aside measurement would favor
    # I2-SPLIT or AUDIT-ALL under real costs.
    c_review, c_audit, c_fn = 5, 1, 15
    erg_low = errors_flagged[is_low_i2].sum()
    erg_high = errors_flagged[~is_low_i2].sum()
    n_low = int(is_low_i2.sum())
    n_high = int((~is_low_i2).sum())

    def cost_i2split(p_low, p_high):
        return n_low * c_audit + (1 - p_low) * erg_low * c_fn + n_high * c_review

    def cost_audit_all_true(p_low, p_high):
        return n_flagged * c_audit + (1 - p_low) * erg_low * c_fn + (1 - p_high) * erg_high * c_fn

    if not np.isnan(p_low_measured) and not np.isnan(p_high_measured):
        ci2 = cost_i2split(p_low_measured, p_high_measured)
        cat = cost_audit_all_true(p_low_measured, p_high_measured)
        p_high_breakeven = 1 - n_high * (c_review - c_audit) / (erg_high * c_fn)
        winner = "I2-SPLIT" if ci2 < cat else ("TIE" if abs(ci2 - cat) < 1e-9 else "AUDIT-ALL")
        print(f"\nUsing MEASURED p_low={p_low_measured:.4f}, p_high={p_high_measured:.4f} in the same cost "
              f"model as triage_cost_experiment.py (C_review={c_review}, C_audit={c_audit}, C_FN={c_fn}):")
        print(f"  I2-SPLIT cost = {ci2:.1f}   AUDIT-ALL(true) cost = {cat:.1f}   -> {winner} wins")
        print(f"  Breakeven p_high for this cost structure = {p_high_breakeven:.5f} "
              f"(measured p_high {'is' if p_high_measured < p_high_breakeven else 'is NOT'} below it)")

    print("\nCAVEAT (stated per this script's docstring): LR is one of the three votes defining I2_hat "
          "itself, so this is not a fully independent audit mechanism -- if I2_hat is high specifically "
          "because LR dissented and LR happens to be right, that mechanically inflates the apparent "
          "p_high here. This measurement should be read as a first, non-circular-free lower-cost proxy, "
          "not as validation of a truly independent audit process.")
