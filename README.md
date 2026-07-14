# NLP Portfolio — Sahar Moghtaderi

My portfolio for the NLP master's course. The grade is based on four deliverables, so I keep
them together in this one repo. It's a solo portfolio — I did all of it myself.

Two of the four are done so far:

**Capstone → [`capstone/`](capstone/)**
"LuV-FaithCheck", a note-grounded faithfulness checker for German vocational reports (LuV).
For each claim in a report it retrieves the relevant case notes and uses natural language
inference (NLI) to decide whether the claim is supported, contradicted, or not mentioned, and
flags the ones a reviewer should check. Full write-up in
[`capstone/REPORT.md`](capstone/REPORT.md); run it with `cd capstone && bash run.sh`.

**Paper presentation → [`presentation/`](presentation/)**
The paper I presented: Poliak et al. (2018), *Hypothesis-Only Baselines in NLI*. It shows that
NLI models can often predict the label from the hypothesis alone (an annotation-artifact
problem), so you should always compare against a hypothesis-only baseline. That's the same
baseline logic I use in the capstone, which is why I chose it.

Still to add: **participation** (tutorial / poetry slam / DAAD event) and **competition** (my score).

## Contributions

Solo project: **Sahar Moghtaderi — 100%** of all four deliverables (presentation,
participation, competition, and capstone).

## A note on the data

The capstone results come from real (anonymised) case notes about actual people, and that data
can't leave the institute — so it is **not** in this repo. I included small synthetic example
data instead, so the code still runs. Details in the capstone README.
