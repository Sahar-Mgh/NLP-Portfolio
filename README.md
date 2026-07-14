# NLP Portfolio — Sahar Moghtaderi

This is my portfolio for the NLP master's course. The grade is based on four deliverables,
so I keep them all together in this one repo. It's a solo portfolio — I did everything myself.

Two of the four are done so far:

**Capstone → [`capstone/`](capstone/)**
My main project, "LuV-FaithCheck". It's a small tool that checks whether the sentences in a
German vocational report are actually backed up by the trainee's case notes, and flags the
ones that aren't. There's a full write-up in [`capstone/REPORT.md`](capstone/REPORT.md), and
you can run it with `cd capstone && bash run.sh`.

**Paper presentation → [`presentation/`](presentation/)**
The paper I presented: Poliak et al. (2018), *Hypothesis-Only Baselines in NLI*. It shows
that NLI models can often guess the answer from just the hypothesis, so you should always
compare against a "hypothesis-only" baseline. That's basically the same idea I use in the
capstone, which is why I chose it.

Still to add: **participation** (tutorial / poetry slam / DAAD event) and **competition** (my score).

## Contributions

Solo project, so this part is simple: **Sahar Moghtaderi — 100%** of all four deliverables
(presentation, participation, competition, and capstone).

## A note on the data

The capstone results come from real (anonymised) case notes about actual people, and that
data isn't allowed to leave the institute — so it is **not** in this repo. I included small
made-up example data instead, just so the code still runs. More on that in the capstone README.
