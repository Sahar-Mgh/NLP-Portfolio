# LuV-FaithCheck

This is my capstone project for the NLP course: a **note-grounded faithfulness checker** for
German vocational reports (*Leistungs- und Verhaltensbeurteilung*, "LuV"). For each sentence
(a "claim") in a report, it retrieves the relevant case notes and uses **natural language
inference (NLI)** to decide whether the claim is **supported**, **contradicted**, or
**not mentioned** — flagging the sentences a reviewer should check, and giving the note it
relied on as a citation plus a confidence score.

It's zero-shot and runs locally on CPU, which matters because the real data is
sensitive and can't leave the institute. Going zero-shot was deliberate: an earlier version of
the project fine-tuned a model (LoRA/PEFT), but it learned a negation shortcut — a
hypothesis-only baseline that never sees the notes did just as well — so I dropped training and
use the labels only for evaluation.

The full write-up — method, experiments, error analysis, references — is in
[REPORT.md](REPORT.md). This README is the short "what is it / how to run it" version.

## About the data

The results in the report come from real, anonymised case notes about actual rehabilitation
participants. That data can't leave the institute, so it is **not** in this repo.

Instead I included small synthetic example files (the `*.mock.*` ones) with the same schema,
so the pipeline still runs if you clone it. The real numbers in the report can only be
reproduced on the real data.

## How to run it

```bash
pip install -r requirements.txt

bash run.sh                                        # Linux / macOS
powershell -ExecutionPolicy Bypass -File run.ps1   # Windows
```

This runs the full pipeline on the synthetic demo data and prints the evaluation table. It's
a small demo (14 claims across 3 fictional trainees), so the numbers won't match the report —
but I hand-built the examples so they demonstrate the findings I discuss in the report (below).
The first run downloads two models from Hugging Face (a few hundred MB), then runs offline.

If you have the real data locally, `REAL=1 bash run.sh` reproduces the report.

### What the demo examples show

I picked the fictional cases so the demo demonstrates the findings, not just executes:

- **`s1_c4`** ("ist nicht unzuverlässig") — the lexical (TF-IDF + negation-cue) baseline is
  fooled by the double negation and flags it; the NLI model handles it correctly.
- **`s3_c2`** (an abstractive claim with a specific date) — no single note entails the whole
  claim, so a genuinely supported claim gets missed. This **granularity mismatch** is the main
  limitation I found (report §6).
- **`s3_c3`** ("bleibt ausgeglichen" vs a note about stress) — a real contradiction on the
  same topic, correctly flagged.
- **`s3_c4`** — an off-topic note with a negation cue that the raw NLI reads as a
  contradiction; the NER entity gate is meant to suppress it (report §5.2).

## Results (on the real data, from the report)

Evaluated on 90 hand-labelled claims. The majority-class baseline gets 0.544 accuracy, so
that's the bar. The metric I care about most is **flag recall** — of the sentences a reviewer
should check (contradicted or not-mentioned), how many the system actually surfaces.

| system | macro-F1 | accuracy | flag recall | contradictions caught |
| --- | :-: | :-: | :-: | :-: |
| majority baseline | ~0.23 | 0.544 | — | — |
| lexical (TF-IDF + negation) | 0.329 | 0.456 | 0.54 | 0 / 3 |
| zero-shot NLI | 0.452 | 0.511 | 0.80 | 3 / 3 |
| NLI + NER gate | 0.478 | 0.533 | 0.80 | 3 / 3 |

Short version: zero-shot NLI clearly beats the lexical baseline on flagging (0.80 vs 0.54
recall) and recovers all three contradictions. Overall accuracy stays near the majority
baseline because the classes are imbalanced and some claims are genuinely hard — I discuss
this honestly in the report.

## How it works

1. **Retrieval** — embed the claim and the trainee's notes with a multilingual Sentence-BERT
   (`paraphrase-multilingual-MiniLM-L12-v2`) and take the top *k = 8* notes by cosine similarity.
2. **NLI scoring** — run a zero-shot multilingual NLI model (`mDeBERTa-v3-xnli`) on each
   (note, claim) pair to get entailment / contradiction / neutral probabilities.
3. **Aggregation** — SummaC-style: the maximum entailment over the retrieved notes is the
   support score and the maximum contradiction is the flag score; thresholds turn these into
   the verdict.
4. **NER entity gate** — a lightweight lexicon/regex NER so an off-topic note can only trigger
   a contradiction if it shares a topic (SKILL/AREA) entity with the claim.

Output per claim: supported / contradicted / not_mentioned, the note that drove the decision
(citation), and a confidence.

## Files

- `faithcheck.py` — the pipeline (retrieval → NLI/lexical scoring → aggregation → gate → verdict)
- `evaluate.py` — scores predictions against the gold labels and compares methods
- `ner.py` — the lexicon/regex NER used for the contradiction gate
- `tune.py` — threshold tuning on a dev/test split
- `build_dataset.py`, `make_worksheet.py`, `labels_from_worksheet.py` — how I built the
  dataset and gold labels from the real files (these need that data to run)
- `data/*.mock.jsonl` — synthetic demo data
- `results/*.mock.*` — demo outputs
- `REPORT.md` — the full write-up

## NLP techniques used (course requirement)

Sentence embeddings (SBERT retrieval), transfer learning (zero-shot NLI, no fine-tuning),
pretrained language models (mDeBERTa / MiniLM), evaluation methods (macro-F1, per-class
scores, majority baseline, dev/test threshold tuning), and named entity recognition (the
entity gate). More detail in the report (§7).
