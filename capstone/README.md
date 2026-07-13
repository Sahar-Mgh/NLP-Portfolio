# LuV-FaithCheck — Note-Grounded Faithfulness Checking for German Vocational Reports

A **zero-shot, locally-run** system that checks whether each sentence in a German
*Leistungs- und Verhaltensbeurteilung* (LuV) report is **grounded in the trainee's case
notes** — flagging unsupported or contradicted claims with a **citation** and a
**confidence score**. It is the "red squiggle" for report writing: retrieve the relevant
notes, run natural-language inference (NLI), and tell a reviewer which sentences to check.

> Runs fully on-premise on a laptop CPU. No training, no data leaves the machine —
> which is the point, because the underlying data is sensitive.

**Full write-up (method, experiments, error analysis, references):** see **[REPORT.md](REPORT.md)**.

---

> ## 🔒 About the data
>
> The reported results were produced on **real, anonymised case-note data about actual
> rehabilitation participants**, which — by design and by data-protection obligation —
> **does not leave the institute** and is therefore **not included in this repository**.
>
> This repo ships small **synthetic (`*.mock.*`) stand-ins** with the same schema so the
> pipeline runs on a clone. The real numbers in [REPORT.md](REPORT.md) can be reproduced
> only on the real data, on the institute's own machines.
>
> `.gitignore` is configured so that any real `data/*.jsonl` or `results/*.jsonl` on a
> local machine is **never tracked or pushed** — only `*.mock.*` files are committed.

---

## Contributions

**Solo project — Sahar Moghtaderi contributed 100% to all components** (idea, dataset
construction, method, implementation, hand-labelling, evaluation, and write-up).

## Quickstart (runs on the bundled synthetic data)

```bash
pip install -r requirements.txt   # CPU is fine; see requirements.txt if torch won't resolve

bash run.sh                       # macOS / Linux  — mock demo
powershell -ExecutionPolicy Bypass -File run.ps1   # Windows — mock demo
```

This runs retrieval → NLI → aggregation → gate on the synthetic data and prints the
evaluation table. It is a small demo (14 claims across 3 fictional trainees), so the numbers
are **not** the REPORT.md numbers — but the cases are hand-built to *exhibit the report's
findings*, not just execute. The first NLI run downloads two models (~a few hundred MB) from
Hugging Face, then runs offline.

**Reproducing the report** (only on the real institute data, held locally): `REAL=1 bash run.sh`.

### What the synthetic demo is built to show

| case (fictional) | what it demonstrates |
| --- | --- |
| `s1_c4` "ist **nicht unzuverlässig**" → supported | NLI beats the lexical baseline's **negation shortcut** (lexical mis-flags the double negative) |
| `s3_c2` a date-bundled abstractive claim → missed | **granularity mismatch**: no single note entails the summary, so a *supported* claim is missed ([REPORT.md §6](REPORT.md)) |
| `s3_c3` "bleibt ausgeglichen" vs a stress note → flagged | a **real contradiction on the same topic** is correctly kept by the gate |
| `s3_c4` off-topic negation note trips raw NLI | the **NER gate tradeoff**: the *hard* gate removes the false alarm, the *soft* gate keeps it ([REPORT.md §5.2](REPORT.md)) |

## Results on the real data (from [REPORT.md](REPORT.md) §5.1)

Evaluated on **90 hand-labelled gold claims**; majority-class baseline accuracy **0.544**.
The headline improvement is **flag recall** (the deployable "needs-review" signal), which is
robust to the class imbalance; the NER gate is a small, honest refinement (see REPORT.md §5).

| system                     | macro-F1  | accuracy | flag recall | true contradictions kept |
| -------------------------- | :-------: | :------: | :---------: | :----------------------: |
| majority baseline          |   ~0.23   |  0.544   |     —       |           —              |
| old lexical (TF-IDF + neg.)|   0.329   |  0.456   |    0.54     |          0 / 3           |
| zero-shot NLI              |   0.452   |  0.511   |    0.80     |          3 / 3           |
| **NLI + NER soft-gate**    | **0.478** |  0.533   |    0.80     |          3 / 3           |

## How it works (one paragraph)

For each report **claim**, embed it and the trainee's notes with a multilingual
Sentence-BERT and retrieve the top *k = 8* notes; score each (note, claim) pair with a
zero-shot multilingual NLI model (`mDeBERTa-v3-xnli`); aggregate SummaC-style (max
entailment = support, max contradiction = flag); and gate contradictions by a lightweight
domain NER so an off-topic note cannot raise a false alarm. Output: `supported` /
`contradicted` / `not_mentioned`, plus the driving note (citation) and a confidence.

## Repository layout

| path | what it is |
| --- | --- |
| `faithcheck.py` | the pipeline: retrieve → NLI/lexical score → aggregate → gate → verdict |
| `evaluate.py` | scores predictions vs. gold labels; multi-method comparison + disagreement table |
| `ner.py` | offline lexicon/regex domain NER used by the contradiction gate |
| `tune.py` | dev/test threshold tuning (avoids threshold cherry-picking) |
| `build_dataset.py`, `make_worksheet.py`, `labels_from_worksheet.py` | how the real dataset and gold labels were built (run against local institute data; provenance) |
| `data/*.mock.jsonl` | synthetic claims / notes / labels so the pipeline runs without real data |
| `results/*.mock.*` | evaluation outputs on the synthetic data |
| `REPORT.md` | full academic report (method, results, error analysis, references) |

> The real `data/*.jsonl`, real `results/*.jsonl`, and the labelling worksheets are held
> locally and git-ignored — see the data note above. The labelling methodology is described
> in [REPORT.md §2](REPORT.md).

## NLP techniques used (course mapping)

Embeddings (SBERT retrieval) · Transfer learning (zero-shot XNLI→LuV) · Language models
(mDeBERTa / MiniLM) · Evaluation methods (macro-F1, per-class, majority baseline, dev/test
tuning) · Named-entity recognition (contradiction gate). Detail in [REPORT.md §7](REPORT.md).
