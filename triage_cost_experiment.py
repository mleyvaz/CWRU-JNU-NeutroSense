# -*- coding: utf-8 -*-
"""
Tests whether using I2_hat (decision disagreement) to route already-flagged
instances between a CHEAP, less-reliable intervention (AUDIT) and an
EXPENSIVE, fully-reliable one (REVIEW) reduces total expected cost compared
to (a) routing everyone to REVIEW, or (b) routing to AUDIT/REVIEW in the
same proportion but RANDOMLY rather than using I2_hat -- i.e., isolating
whether I2_hat specifically, not just "sometimes using a cheaper option",
drives any savings.

REVISION NOTE: a first version of this experiment defined AUDIT as a flat,
always-successful cheap action, which made "audit everything" the trivial
global optimum whenever C_audit < C_review, regardless of I1_hat or
I2_hat -- a degenerate result that did not test the framework at all. This
version instead ties AUDIT's SUCCESS PROBABILITY to I2_hat: audits
automated, lightweight cross-checks are assumed more likely to resolve a
flagged case correctly when the base learners already agree (low I2_hat)
than when they genuinely disagree (high I2_hat), which requires human
judgment to adjudicate. This is a stated, illustrative modeling assumption,
not a measured quantity -- the point is to test whether, GIVEN this kind of
asymmetry (audit is unreliable specifically where models disagree), routing
by I2_hat captures it better than routing at random.

Population: instances FLAGGED for intervention by I1_hat alone (I1_hat >
tau1, the same "ambiguous signal" criterion a single-score policy would
use) are not accepted automatically in any of the three policies compared;
only how they are split between AUDIT and REVIEW differs. The accept-zone
(I1_hat <= tau1) is identical and unaffected across all three policies, so
comparisons isolate the value of I2_hat-based routing specifically.

Policies compared, all applied to the SAME flagged set (I1_hat > tau1):
  ALL-REVIEW      : every flagged instance -> REVIEW (cost C_review, always
                    resolves the case, i.e. incurs no residual error cost).
  RANDOM-SPLIT     : flagged instances split into AUDIT/REVIEW in the same
                    global proportion as I2-SPLIT below, but the assignment
                    is a random permutation (does not use I2_hat) -- control
                    condition isolating whether I2_hat's specific groupings
                    matter, not just "using cheaper audit sometimes".
  I2-SPLIT         : flagged instances with I2_hat <= tau2 (base learners
                    mostly agree) -> AUDIT (cost C_audit, succeeds with
                    probability p_audit_agree); flagged instances with
                    I2_hat > tau2 (base learners disagree) -> REVIEW (cost
                    C_review, always resolves).
For AUDIT, if it does not succeed (probability 1-p_audit_*), the case is
NOT corrected: it still costs C_audit (the audit attempt itself), plus
C_FN if the underlying ensemble prediction for that instance was actually
wrong (an undetected error slips through the failed audit).
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


def expected_cost_all_review(n_flagged, c_review):
    return n_flagged * c_review


def expected_cost_split(is_low_i2_group, errors_flagged, c_audit, c_review, p_audit_success):
    """is_low_i2_group: boolean array (True = assigned to AUDIT for this policy)."""
    audit_mask = is_low_i2_group
    review_mask = ~is_low_i2_group
    # AUDIT group: pay C_audit always; pay C_FN_extra (handled by caller via errors) if audit fails AND instance was an error.
    n_audit = audit_mask.sum()
    n_review = review_mask.sum()
    audit_errors = errors_flagged[audit_mask].sum()
    cost_audit = n_audit * c_audit + (1 - p_audit_success) * audit_errors * C_FN_GLOBAL[0]
    cost_review = n_review * c_review
    return cost_audit + cost_review, n_audit, n_review, audit_errors


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

    entropy = -np.sum(P_avg * np.log(P_avg + 1e-12), axis=1) / np.log(P_avg.shape[1])
    votes = np.stack([np.argmax(P_rf, axis=1), np.argmax(P_xgb, axis=1), np.argmax(P_lr, axis=1)], axis=1)
    agree = (votes == y_pred[:, None]).sum(axis=1)
    vote_dis = 1.0 - agree / 3.0

    # Flagged set: same I1_hat-based criterion any single-score policy would use.
    # Use the median split from the manuscript's own Section 4.3 protocol.
    tau1 = np.median(entropy)
    flagged = entropy > tau1
    n_flagged = flagged.sum()
    errors_flagged = errors[flagged]
    vote_dis_flagged = vote_dis[flagged]
    print(f"\nFlagged (I1_hat > median): N={n_flagged} of {len(entropy)} "
          f"({n_flagged/len(entropy)*100:.1f}%), error rate among flagged = {errors_flagged.mean()*100:.1f}%")

    is_low_i2 = vote_dis_flagged <= np.median(vote_dis_flagged)
    print(f"Within flagged set: {is_low_i2.sum()} have I2_hat <= median (models mostly agree), "
          f"{(~is_low_i2).sum()} have I2_hat > median (models disagree)")
    print(f"  Error rate | low I2 (candidates for cheap audit):  {errors_flagged[is_low_i2].mean()*100:.1f}%")
    print(f"  Error rate | high I2 (routed to full review):      {errors_flagged[~is_low_i2].mean()*100:.1f}%")

    scenarios = [
        # (name, C_review, C_audit, C_FN, p_audit_success_low_I2, p_audit_success_high_I2)
        ("S1: audit fairly reliable when models agree", 5, 1, 15, 0.90, 0.90),
        ("S2: audit much less reliable under disagreement", 5, 1, 15, 0.90, 0.30),
        ("S3: same as S2, cheaper audit",                  5, 1, 15, 0.95, 0.30),
        ("S4: no reliability gap (control)",                5, 1, 15, 0.70, 0.70),
    ]

    rng = np.random.RandomState(42)
    print(f"\n{'Scenario':46s} {'ALL-REVIEW':>12s} {'RANDOM-SPLIT':>14s} {'I2-SPLIT':>10s} {'Savings vs review':>18s} {'Savings vs random':>18s}")
    for name, c_review, c_audit, c_fn, p_low, p_high in scenarios:
        C_FN_GLOBAL = [c_fn]
        cost_all_review = expected_cost_all_review(n_flagged, c_review)

        # I2-SPLIT: audit success probability depends on which I2 group (this IS the mechanism being tested)
        cost_i2split_audit = is_low_i2.sum() * c_audit + (1 - p_low) * errors_flagged[is_low_i2].sum() * c_fn
        cost_i2split_review = (~is_low_i2).sum() * c_review
        cost_i2split = cost_i2split_audit + cost_i2split_review

        # RANDOM-SPLIT control: same proportion audited, but WHICH instances are audited is random,
        # so audit success probability is the flagged-set AVERAGE of the two group rates, weighted by
        # how many of each true I2 group land in the (randomly chosen) audit bucket -- averaged over
        # many random permutations for a stable expectation.
        n_audit_target = is_low_i2.sum()
        random_costs = []
        for _ in range(2000):
            perm = rng.permutation(n_flagged)
            audit_idx = perm[:n_audit_target]
            audit_mask = np.zeros(n_flagged, dtype=bool)
            audit_mask[audit_idx] = True
            # success probability for each randomly-audited instance depends on ITS OWN true I2 group
            p_success_per_instance = np.where(is_low_i2[audit_mask], p_low, p_high)
            fail_mask = rng.random_sample(audit_mask.sum()) > p_success_per_instance
            audit_fail_errors = errors_flagged[audit_mask][fail_mask].sum()
            c = audit_mask.sum() * c_audit + audit_fail_errors * c_fn + (~audit_mask).sum() * c_review
            random_costs.append(c)
        cost_random = float(np.mean(random_costs))

        sav_vs_review = (cost_all_review - cost_i2split) / cost_all_review * 100
        sav_vs_random = (cost_random - cost_i2split) / cost_random * 100
        print(f"{name:46s} {cost_all_review:12.1f} {cost_random:14.1f} {cost_i2split:10.1f} "
              f"{sav_vs_review:17.2f}% {sav_vs_random:17.2f}%")

    print(f"\n(All costs are expected total cost over the {n_flagged} flagged JNU test instances, "
          f"arbitrary relative units; RANDOM-SPLIT averaged over 2000 random permutations.)")
