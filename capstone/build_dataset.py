"""
build_dataset.py -- convert Sahar's real LuV data into the faithcheck schema.
=============================================================================
Reads the 5 per-participant claims files and 5 per-participant notes files and
writes the two JSONL files the pipeline expects:

  data/claims.jsonl : {claim_id, student_id, area, claim}
  data/notes.jsonl  : {note_id, student_id, date, author, title, text}

Linkage: claims carry student_id "TN-0000004540"; notes carry tn "0000004540",
so a note's student_id = "TN-" + tn.  We verify every claim's student has notes.

Run: python build_dataset.py
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CLAIMS_DIR = r"C:\0. Thesis\final\Cursor\claims"
NOTES_DIR = r"C:\0. Thesis\final\Cursor\notes\clean notes\cleaned_json\clean"
OUT = Path(__file__).resolve().parent / "data"   # writes next to this script (capstone/data)


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                       encoding="utf-8")


def main():
    # ---- claims ----
    claims, claim_students = [], Counter()
    for f in sorted(glob.glob(os.path.join(CLAIMS_DIR, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        sid = d["student_id"]
        for c in d["claims"]:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            claims.append({"claim_id": c["id"], "student_id": sid,
                           "area": c.get("section") or "", "claim": txt})
            claim_students[sid] += 1

    # ---- notes ----
    notes, note_students = [], Counter()
    for f in sorted(glob.glob(os.path.join(NOTES_DIR, "*.json"))):
        arr = json.load(open(f, encoding="utf-8"))
        for n in arr:
            txt = (n.get("text") or "").strip()
            if not txt:
                continue
            sid = "TN-" + str(n["tn"])
            notes.append({"note_id": n["id"], "student_id": sid,
                          "date": n.get("date") or n.get("date_original") or "",
                          "author": n.get("author") or "", "title": n.get("title") or "",
                          "text": txt})
            note_students[sid] += 1

    write_jsonl(OUT / "claims.jsonl", claims)
    write_jsonl(OUT / "notes.jsonl", notes)

    # ---- report + linkage check ----
    print(f"claims.jsonl : {len(claims)} claims across {len(claim_students)} students")
    print(f"notes.jsonl  : {len(notes)} note-sentences across {len(note_students)} students\n")
    print(f"{'student_id':16s} {'claims':>7s} {'notes':>7s}")
    for sid in sorted(claim_students):
        print(f"{sid:16s} {claim_students[sid]:7d} {note_students.get(sid, 0):7d}")

    missing = [s for s in claim_students if s not in note_students]
    print("\nlinkage:", "OK - every claim's student has notes" if not missing
          else f"WARNING - students with claims but NO notes: {missing}")

    areas = Counter(c["area"] for c in claims)
    print(f"\nclaim 'section'/area values ({len(areas)} distinct):")
    for a, n in areas.most_common():
        print(f"  {n:4d}  {a}")


if __name__ == "__main__":
    main()
