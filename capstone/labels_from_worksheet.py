"""
Turn a filled labeling worksheet into labels.jsonl.

After the `label` column is filled (supported / contradicted / not_mentioned), this
reads the worksheet back and writes the labels.jsonl that evaluate.py expects.

Run: python labels_from_worksheet.py --worksheet labeling/worksheet.mock.xlsx --out data/labels.mock.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VALID = {"supported", "contradicted", "not_mentioned"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", required=True)
    ap.add_argument("--out", default="data/labels.jsonl")
    args = ap.parse_args()

    p = Path(args.worksheet)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)

    rows, skipped = [], 0
    for _, r in df.iterrows():
        lab = str(r.get("label", "")).strip().lower()
        if lab not in VALID:
            skipped += 1
            continue
        rows.append({"claim_id": str(r["claim_id"]), "label": lab})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                              encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} labels  ({skipped} rows skipped as unlabeled/invalid)")
    if rows:
        dist = {l: sum(x["label"] == l for x in rows) for l in sorted(VALID)}
        print("label distribution:", dist)


if __name__ == "__main__":
    main()
