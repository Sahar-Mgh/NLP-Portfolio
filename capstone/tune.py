"""
Tuned comparison of systems on the labelled set.

For each predictions file, re-derives the verdict from the stored
support/contradiction scores over a grid of thresholds, picks the
(tau_sup, tau_con) that maximise macro-F1 on a dev split, and reports
macro-F1/accuracy on the held-out test split. This keeps the thresholds
from being tuned on the same data they're scored on. Also prints the
number at the default threshold (0.55) for reference.

Verdict rule (matches faithcheck.judge_claim):
    if con >= tau_con and con >= sup -> contradicted
    elif sup >= tau_sup             -> supported
    else                            -> not_mentioned

Run:
  python tune.py --labels data/labels.jsonl \
    --preds results/preds_nli.jsonl results/preds_nli_improved.jsonl results/preds_lexical.jsonl \
    --names zero-shot-NLI improved-NLI old-lexical
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = ["supported", "contradicted", "not_mentioned"]
GRID = [round(x, 2) for x in np.arange(0.40, 0.86, 0.05)]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def verdict(sup, con, tau_sup, tau_con):
    if con >= tau_con and con >= sup:
        return "contradicted"
    if sup >= tau_sup:
        return "supported"
    return "not_mentioned"


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def evaluate_system(rows, gold, dev_ids, test_ids):
    def preds_at(ids, ts, tc):
        return [verdict(rows[i]["support_score"], rows[i]["contradiction_score"], ts, tc) for i in ids]

    def truth(ids):
        return [gold[i] for i in ids]

    # tune on dev
    best, best_f1 = (0.55, 0.55), -1
    for ts in GRID:
        for tc in GRID:
            f = macro_f1(truth(dev_ids), preds_at(dev_ids, ts, tc))
            if f > best_f1:
                best_f1, best = f, (ts, tc)
    ts, tc = best
    yt, yp = truth(test_ids), preds_at(test_ids, ts, tc)
    yt_d, yp_d = truth(test_ids), preds_at(test_ids, 0.55, 0.55)
    return {
        "best_tau": best, "dev_macro_f1": round(best_f1, 3),
        "test_acc_tuned": round(accuracy_score(yt, yp), 3),
        "test_macro_f1_tuned": round(macro_f1(yt, yp), 3),
        "test_acc_default": round(accuracy_score(yt_d, yp_d), 3),
        "test_macro_f1_default": round(macro_f1(yt_d, yp_d), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--dev_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = args.names or [Path(p).stem.replace("preds_", "") for p in args.preds]
    gold = {r["claim_id"]: r["label"] for r in read_jsonl(args.labels) if r.get("label") in LABELS}
    systems = {nm: {r["claim_id"]: r for r in read_jsonl(p)} for nm, p in zip(names, args.preds)}

    ids = [i for i in gold if all(i in s for s in systems.values())]
    rng = np.random.default_rng(args.seed)
    ids = list(rng.permutation(ids))
    n_dev = int(args.dev_frac * len(ids))
    dev_ids, test_ids = ids[:n_dev], ids[n_dev:]
    maj = max(sum(gold[i] == l for i in test_ids) for l in LABELS) / max(len(test_ids), 1)

    print(f"labelled={len(ids)}  dev={len(dev_ids)}  test={len(test_ids)}  "
          f"majority-baseline test acc={maj:.3f}\n")
    print(f"{'system':16s} {'best_tau':>14s} {'devF1':>6s} | {'testAcc':>8s} {'testF1':>7s} "
          f"| {'defAcc':>7s} {'defF1':>6s}")
    for nm in names:
        r = evaluate_system(systems[nm], gold, dev_ids, test_ids)
        print(f"{nm:16s} {str(r['best_tau']):>14s} {r['dev_macro_f1']:6.3f} | "
              f"{r['test_acc_tuned']:8.3f} {r['test_macro_f1_tuned']:7.3f} | "
              f"{r['test_acc_default']:7.3f} {r['test_macro_f1_default']:6.3f}")
    print("\n(tuned = thresholds chosen on dev; default = tau 0.55. Small n -> read as indicative.)")


if __name__ == "__main__":
    main()
