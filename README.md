# NLP Portfolio — Sahar Moqtaderi

Master's NLP course portfolio. Per the course scheme, the portfolio bundles the four
deliverables in a single repository; this top-level README records the contribution to
each component (as the course requires).

## Contributions

Solo portfolio — **Sahar Moqtaderi contributed 100%** to every deliverable below.
*(If any deliverable was produced in a team, adjust that row.)*

| # | Deliverable | Contribution | Status |
|---|---|---|---|
| 1 | Paper presentation | Sahar Moqtaderi — 100% | _to add_ |
| 2 | Course participation (tutorial, poetry slam, DAAD event) | Sahar Moqtaderi — 100% | _to add_ |
| 3 | Competition result | Sahar Moqtaderi — 100% | _to add_ |
| 4 | **Capstone — LuV-FaithCheck** | Sahar Moqtaderi — 100% | ✅ [`capstone/`](capstone/) |

## Deliverables

### 4. Capstone — LuV-FaithCheck → [`capstone/`](capstone/)

A zero-shot, locally-run **faithfulness checker** that flags whether sentences in a German
vocational report (*Leistungs- und Verhaltensbeurteilung*) are grounded in the trainee's case
notes: retrieval → multilingual NLI → entity-gated aggregation, with a citation and a
confidence score. Full details in [`capstone/README.md`](capstone/README.md) and the write-up
[`capstone/REPORT.md`](capstone/REPORT.md).

```bash
cd capstone
pip install -r requirements.txt
bash run.sh          # runs the pipeline on the bundled synthetic demo data
```

> ⚠️ The capstone ships **synthetic demo data only**. The real (sensitive) participant data
> is kept off GitHub and never committed — see the capstone README for the reasoning.

### 1–3. Presentation · Participation · Competition

_Add each deliverable here as it is ready — e.g. `presentation/` (slides or PDF),
`participation/` (evidence of the tutorial / poetry slam / DAAD event), `competition/`
(final score or rank) — and fill in the Status column above._
