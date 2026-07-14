# LuV-FaithCheck

This is my capstone project for the NLP course. It's a small tool that checks whether the
sentences in a German vocational report (a *Leistungs- und Verhaltensbeurteilung*, or "LuV")
are actually supported by the trainee's case notes. It flags the sentences that look
unsupported or contradicted, and for each one it points to the note it used and gives a
confidence score. Basically a "is this claim actually backed up?" checker for report writing.

It runs locally on a normal laptop (no GPU needed) and doesn't train anything, which matters
because the real data is sensitive and can't leave the institute.

The full write-up — method, experiments, error analysis, references — is in
[REPORT.md](REPORT.md). This README is just the short "what is it / how do I run it" version.

## About the data (please read)

The results in the report come from real, anonymised case notes about actual rehabilitation
participants. That data isn't allowed to leave the institute, so it is **not** in this repo.

Instead I included small **made-up** example files (the `*.mock.*` ones) with the same format,
so the code still runs if you clone it. The real numbers in the report can only be reproduced
on the real data, on the institute's own machines. The `.gitignore` is set up so the real data
can never get committed by accident — only the made-up files are tracked.

## How to run it

```bash
pip install -r requirements.txt

bash run.sh                                        # Linux / macOS
powershell -ExecutionPolicy Bypass -File run.ps1   # Windows
```

This runs the whole thing on the made-up demo data and prints an evaluation table. It's a
tiny demo (14 example claims across 3 fictional trainees), so the numbers won't match the
report — but I made up the examples on purpose so they show the things I talk about in the
report (see below). The first run downloads two models from Hugging Face (a few hundred MB),
then works offline after that.

If you actually have the real data locally, `REAL=1 bash run.sh` reproduces the report.

### What the demo examples are meant to show

I picked the fictional cases so the demo actually shows my findings instead of just running:

- **`s1_c4`** ("ist nicht unzuverlässig") — the old keyword baseline gets tricked by the
  double negative and flags it; the NLI model gets it right.
- **`s3_c2`** (a claim with a specific date) — no single note states the whole thing, so the
  tool misses a claim that's actually supported. This is the main limitation I found (report §6).
- **`s3_c3`** ("stays calm under stress" vs a note about him getting stressed) — a real
  contradiction on the same topic, correctly flagged.
- **`s3_c4`** — an off-topic note with a negative word that fools the raw NLI; the NER step
  is meant to catch this (report §5.2).

## Results (on the real data, from the report)

Tested on 90 hand-labelled examples. If you just always guess the most common label you get
0.544 accuracy, so that's the bar to beat. The number I care about most is **flag recall** —
out of the sentences a reviewer really should double-check, how many the tool actually catches.

| system | macro-F1 | accuracy | flag recall | contradictions caught |
| --- | :-: | :-: | :-: | :-: |
| always-guess-majority | ~0.23 | 0.544 | — | — |
| old keyword baseline | 0.329 | 0.456 | 0.54 | 0 / 3 |
| zero-shot NLI | 0.452 | 0.511 | 0.80 | 3 / 3 |
| NLI + NER gate | 0.478 | 0.533 | 0.80 | 3 / 3 |

Short version: the NLI approach clearly beats the old keyword one at flagging (0.80 vs 0.54
recall). The plain accuracy stays near the baseline because the data is imbalanced and some
claims are genuinely hard — I'm honest about that in the report.

## How it works (short version)

For each claim in the report: embed it and the trainee's notes and take the 8 most similar
notes; run a multilingual NLI model on each (note, claim) pair; use the strongest entailment
as "support" and the strongest contradiction as a "flag"; and use a small keyword-based NER
step so an off-topic note can't set off a false alarm. Output per claim: supported /
contradicted / not_mentioned, plus which note it used and a confidence.

## Files

- `faithcheck.py` — the main pipeline (retrieve → score → aggregate → gate → verdict)
- `evaluate.py` — scores the predictions against the labels and compares methods
- `ner.py` — the little keyword NER for the topic gate
- `tune.py` — threshold tuning on a dev/test split
- `build_dataset.py`, `make_worksheet.py`, `labels_from_worksheet.py` — how I built the
  dataset and labels from the real files (these need that data to run)
- `data/*.mock.jsonl` — the made-up demo data
- `results/*.mock.*` — demo outputs
- `REPORT.md` — the full write-up

## NLP methods I used (for the course requirement)

Sentence embeddings (for retrieval), transfer learning (zero-shot NLI with no fine-tuning),
pretrained language models (mDeBERTa / MiniLM), a proper evaluation (macro-F1, per-class
scores, baselines, threshold tuning), and named-entity recognition (the topic gate). More
detail in the report (§7).
