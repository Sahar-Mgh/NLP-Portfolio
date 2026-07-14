"""
Build a labeling worksheet for grading the grounding checker by hand.

Reads claims.jsonl and notes.jsonl, retrieves each claim's most relevant notes,
and writes an Excel file with one row per claim:

    claim_id | student_id | area | claim | note_1 .. note_k | label

The label column is left blank. Open the file in Excel and fill it with one of
supported / contradicted / not_mentioned; the relevant notes sit next to each
claim so labeling is quick. A second "guide" sheet lists the label definitions.
Then labels_from_worksheet.py turns the filled sheet into labels.jsonl.

--sample N takes a stratified sample of N claims (proportional per student,
seeded) so you can label a representative subset instead of every claim.

The worksheet does not show the model's prediction: labels have to be
independent of the system being graded, otherwise the evaluation is circular.

Run:
    python make_worksheet.py --claims data/claims.jsonl --notes data/notes.jsonl --out labeling/worksheet.xlsx --sample 90 --k 6
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from faithcheck import Retriever, read_jsonl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GUIDE = [
    ("supported", "The notes contain evidence that BACKS the claim (claim is true per the notes)."),
    ("contradicted", "The notes contain evidence that CONFLICTS with the claim (claim is false per the notes)."),
    ("not_mentioned", "The notes don't say anything that settles it either way."),
    ("", ""),
    ("HOW", "Read the claim, then the notes beside it. Judge FROM THE NOTES, not outside knowledge."),
    ("TIP", "If none of the shown notes settle it, it's usually 'not_mentioned'. Leave a row blank to skip it."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--out", default="labeling/worksheet.xlsx")
    ap.add_argument("--k", type=int, default=6, help="notes shown per claim")
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (0 = all claims)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    claims = read_jsonl(args.claims)
    notes = read_jsonl(args.notes)
    by_student_notes = defaultdict(list)
    for n in notes:
        by_student_notes[n["student_id"]].append(n)

    # stratified sample (proportional per student)
    if args.sample and args.sample < len(claims):
        rnd = random.Random(args.seed)
        by_student_claims = defaultdict(list)
        for c in claims:
            by_student_claims[c["student_id"]].append(c)
        total = len(claims)
        sample = []
        for sid, cl in by_student_claims.items():
            take = max(1, round(args.sample * len(cl) / total))
            rnd.shuffle(cl)
            sample += cl[:take]
        rnd.shuffle(sample)
        claims = sample[:args.sample]

    # retrieval (precompute note embeddings once per student)
    retr = Retriever()
    index = {}
    if retr.sbert:
        for sid, ns in by_student_notes.items():
            index[sid] = retr.encode([n["text"] for n in ns])

    rows = []
    for c in claims:
        pool = by_student_notes.get(c["student_id"], [])
        order, _ = retr.rank(c["claim"], [n["text"] for n in pool], doc_emb=index.get(c["student_id"]))
        row = {"claim_id": c["claim_id"], "student_id": c.get("student_id"),
               "area": c.get("area", ""), "claim": c["claim"]}
        for rank, idx in enumerate(order[:args.k], 1):
            n = pool[idx]
            row[f"note_{rank}"] = f'[{n.get("date","")} · {n.get("title","")}] {n.get("text","")}'
        row["label"] = ""      # left blank for the labeler to fill: supported / contradicted / not_mentioned
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="labeling")
            pd.DataFrame(GUIDE, columns=["label", "meaning"]).to_excel(xw, index=False, sheet_name="guide")
        wrote = out
    except Exception as exc:
        wrote = out.with_suffix(".csv")
        df.to_csv(wrote, index=False, encoding="utf-8-sig")
        print(f"[info] xlsx engine unavailable ({exc.__class__.__name__}); wrote CSV instead.")

    by_stud = df["student_id"].value_counts().to_dict()
    print(f"wrote {wrote}  ({len(df)} claims to label)")
    print(f"per-student: {by_stud}")
    print("columns:", list(df.columns))
    print("\nOpen it in Excel, fill the 'label' column, then run labels_from_worksheet.py.")


if __name__ == "__main__":
    main()
