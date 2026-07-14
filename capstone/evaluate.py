"""
Score faithcheck predictions against human labels, and compare methods.

Pass one or more predictions files to get a side-by-side table:
  - per method: accuracy, macro-F1, per-class precision/recall/F1, confusion
    matrix, and precision/recall on the binary "needs_review" flag
  - majority-class baseline accuracy
  - with exactly two methods: a table of the claims where they disagree, showing
    the gold label and which method got it right

Inputs: one or more predictions.jsonl files (from faithcheck.py) and a
labels.jsonl of {claim_id, label}.

Usage:
  python evaluate.py --preds results/preds_nli.mock.jsonl results/preds_lexical.mock.jsonl \
                     --names zero-shot-NLI lexical --labels data/labels.mock.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_recall_fscore_support)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = ["supported", "contradicted", "not_mentioned"]


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def metrics_for(y_true, y_pred, y_bin_true, y_bin_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_bin_true, y_bin_pred, labels=["needs_review"], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "flag_p": float(p[0]), "flag_r": float(r[0]), "flag_f1": float(f[0]),
        "report": rep,
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = args.names or [Path(p).stem.replace("preds_", "").replace(".mock", "") for p in args.preds]
    if len(names) != len(args.preds):
        sys.exit("--names must match the number of --preds files")

    preds_by_method = {nm: {r["claim_id"]: r for r in read_jsonl(p)}
                       for nm, p in zip(names, args.preds)}
    gold = {r["claim_id"]: r["label"] for r in read_jsonl(args.labels)}
    ids = [i for i in gold if all(i in pm for pm in preds_by_method.values())]
    if not ids:
        sys.exit("No overlap between all preds files and labels on claim_id.")

    y_true = [gold[i] for i in ids]
    y_bin_true = ["grounded" if l == "supported" else "needs_review" for l in y_true]
    maj = max(y_true.count(l) for l in set(y_true)) / len(y_true)

    results = {}
    for nm, pm in preds_by_method.items():
        y_pred = [pm[i]["verdict"] for i in ids]
        y_bin_pred = [pm[i]["binary"] for i in ids]
        results[nm] = metrics_for(y_true, y_pred, y_bin_true, y_bin_pred)

    # comparison table
    print(f"=== Comparison  (n={len(ids)}, majority-baseline acc={maj:.3f}) ===")
    print(f"{'method':18s} {'acc':>6s} {'macroF1':>8s} {'flag_P':>7s} {'flag_R':>7s} {'flag_F1':>8s}")
    for nm in names:
        m = results[nm]
        print(f"{nm:18s} {m['accuracy']:6.3f} {m['macro_f1']:8.3f} "
              f"{m['flag_p']:7.3f} {m['flag_r']:7.3f} {m['flag_f1']:8.3f}")

    # per-method per-class detail
    for nm in names:
        m = results[nm]
        print(f"\n--- {nm}: per-class ---")
        print(f"{'class':14s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>4s}")
        for l in LABELS:
            c = m["report"][l]
            print(f"{l:14s} {c['precision']:6.3f} {c['recall']:6.3f} {c['f1-score']:6.3f} {int(c['support']):4d}")
        print("confusion (rows=true, cols=pred):", LABELS)
        for l, row in zip(LABELS, m["confusion"]):
            print(f"  {l:14s} {row}")

    # disagreements (only when exactly 2 methods)
    if len(names) == 2:
        a, b = names
        print(f"\n=== Disagreements: {a} vs {b} ===")
        print(f"{'claim_id':10s} {'gold':14s} {a:>16s} {b:>16s}   who_is_right")
        any_diff = False
        for i in ids:
            va, vb = preds_by_method[a][i]["verdict"], preds_by_method[b][i]["verdict"]
            if va == vb:
                continue
            any_diff = True
            ga = "ok" if va == gold[i] else "X"
            gb = "ok" if vb == gold[i] else "X"
            winner = a if va == gold[i] else (b if vb == gold[i] else "neither")
            print(f"{i:10s} {gold[i]:14s} {va+' ['+ga+']':>16s} {vb+' ['+gb+']':>16s}   {winner}")
        if not any_diff:
            print("  (methods agree on every claim)")

    if args.out:
        summary = {"n": len(ids), "majority_baseline_acc": round(maj, 3),
                   "methods": {nm: {"accuracy": round(results[nm]["accuracy"], 3),
                                    "macro_f1": round(results[nm]["macro_f1"], 3),
                                    "flag_precision": round(results[nm]["flag_p"], 3),
                                    "flag_recall": round(results[nm]["flag_r"], 3),
                                    "confusion": results[nm]["confusion"]} for nm in names}}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
