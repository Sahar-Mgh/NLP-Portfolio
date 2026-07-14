# LuV-FaithCheck: Note-Grounded Faithfulness Checking for German Vocational Reports

**NLP course capstone — Sahar Moghtaderi**

> A zero-shot, locally-run system that checks whether sentences in a German
> *Leistungs- und Verhaltensbeurteilung* (LuV) report are grounded in the trainee's
> case notes, flagging unsupported or contradicted claims with a citation and a
> confidence score.

---

## 1. Motivation and task

The case managers write LuV reports that summarise a trainee's
progress, and those summary sentences should be *faithful* to the underlying case
notes. Manually cross-checking every report sentence against hundreds of notes is
infeasible, so the goal is an assistant that flags sentences a reviewer should look
at — the "red squiggle" of the thesis — while running **fully on-premise** (the data
is sensitive and may not leave the institute).

I frame this as **grounded fact-checking / Natural Language Inference (NLI)**. For a
report **claim** *c* and a candidate **note** *e* (the evidence), the system decides:

| label | NLI relation | meaning |
|---|---|---|
| **supported** | entailment | the notes back the claim |
| **contradicted** | contradiction | the notes conflict with the claim (flag) |
| **not_mentioned** | neutral | the notes do not settle the claim |

This is the same task family as FEVER (Thorne et al., 2018) and summarisation
faithfulness (SummaC, Laban et al., 2022), applied to a novel domain: German
vocational-rehabilitation reports.

## 2. Data

The dataset is built from **5 real, anonymised participants** by `build_dataset.py`,
which links each participant's claims file to their notes file via the trainee ID:

- **398 report claims** (one competence-area sentence each; 16 distinct ICF areas
  such as *schulische Basiskompetenzen*, *personale Kompetenzen*).
- **5,152 sentence-split case notes** (staff observations, each with date, author,
  title). Notes are pre-segmented into sentences, which matters because NLI models
  operate best at sentence granularity (Laban et al., 2022).

For evaluation I hand-labelled a **stratified 90-claim sample** (proportional across
the 5 participants) with the three labels. The gold distribution is **49 supported /
38 not_mentioned / 3 contradicted** — contradictions are genuinely rare in real LuV
data, so the **majority-class baseline accuracy is 0.544**. Labels were assigned from
the notes only (not outside knowledge), and independently of any model output to keep
the evaluation non-circular. The sample was labelled by just me, so inter-annotator agreement is not measured
(see §9).

## 3. Method

**Why zero-shot rather than a trained model.** An earlier version of this project included
both a lexical baseline — the TF-IDF + negation-cue heuristic that appears here as *old
lexical* (§5.1) — and a fine-tuned model (LoRA/PEFT). The fine-tuned model turned out to
exploit an annotation artifact instead of genuinely weighing claim against evidence: a
*hypothesis-only* baseline that never sees the notes matched it and recovered every
contradiction, so the model was riding a negation shortcut — the Poliak et al. (2018)
phenomenon discussed in §8. With only 90 labelled claims (3 contradictions) there is in any
case almost no signal to fine-tune on without overfitting. I therefore drop training entirely
and transfer an XNLI-pretrained NLI model zero-shot, using the labelled set only for
evaluation and threshold tuning — never for training.

The pipeline (`faithcheck.py`) is training-free and runs on a laptop CPU:

1. **Retrieval (sentence embeddings).** For each claim, I embed the claim and the
   trainee's notes with a multilingual Sentence-BERT
   (`paraphrase-multilingual-MiniLM-L12-v2`) and retrieve the top *k = 8* notes by
   cosine similarity. Note embeddings are pre-computed once per trainee so the method
   scales to thousands of notes.
2. **Scoring (zero-shot NLI / transfer learning).** Each (note, claim) pair is scored
   by a multilingual NLI transformer,
   `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, pretrained on XNLI and
   applied **zero-shot** — I transfer its entailment/neutral/contradiction knowledge
   to the LuV domain without fine-tuning.
3. **Aggregation (SummaC-style).** Following SummaC (Laban et al., 2022), I aggregate
   over the retrieved notes: the claim's *support score* is the maximum entailment
   probability, and its *contradiction score* is the maximum contradiction
   probability. A verdict rule with thresholds τ then assigns supported /
   contradicted / not_mentioned, together with the note that drove the decision (the
   **citation**) and a confidence value.
4. **NER entity-gate (Named Entity Recognition).** A note should only be able to
   *contradict* a claim if it is about the same topic. An offline domain NER
   (`ner.py`, lexicon + regex, no model download) tags SKILL/AREA/PERSON/DATE/MEASURE
   entities. Because retrieval is scoped per trainee, PERSON is shared by nearly every
   pair and is useless for gating, so the gate uses **SKILL/AREA** entities only: a
   contradiction is only counted if the note shares a topical entity with the claim.

I compare three systems, all through the identical retrieve→aggregate harness so the
comparison isolates the scorer:

- **old lexical** — the previous project's TF-IDF-similarity + negation-cue heuristic;
- **zero-shot NLI** — steps 1–3 above;
- **NLI + NER soft-gate** — adds step 4 (see §5.2 for the soft variant).

## 4. Experimental setup

Metrics: **accuracy**, **macro-F1** (the primary metric, since the task is
imbalanced), per-class precision/recall/F1, and a **binary flag** metric — precision
and recall of "needs-review" (contradicted ∪ not_mentioned), which is the deployable
signal for a reviewer. Baselines: the majority class (0.544) and the old lexical
system. Thresholds τ are reported both at a fixed default (0.55) and **tuned on a
50/50 dev/test split** (`tune.py`) to avoid threshold cherry-picking. Everything runs
locally on CPU; no data leaves the machine.

## 5. Results

### 5.1 Main comparison (n = 90 gold labels)

| system | macro-F1 | accuracy | flag recall | true-contradictions kept |
|---|---|---|---|---|
| majority baseline | ~0.23 | 0.544 | — | — |
| old lexical | 0.329 | 0.456 | 0.54 | 0 / 3 |
| zero-shot NLI | 0.452 | 0.511 | 0.80 | 3 / 3 |
| **NLI + NER soft-gate** | **0.478** | 0.533 | 0.80 | 3 / 3 |

The zero-shot NLI system **outperforms the previous lexical baseline** on every headline
metric — macro-F1 0.45 vs 0.33, flag recall 0.80 vs 0.54, and it recovers all three gold
contradictions where the lexical model recovers none. The **robust** part of this gain does
not depend on the rare contradiction class: flag recall rises 0.54 → 0.80 (on the 41
needs-review claims) and not_mentioned F1 rises 0.42 → 0.52 (on 38 claims). The macro-F1 gap
is genuine but should be read with care — most of it (≈70%) comes from the contradicted
class, where only 3 gold instances make that single F1 component high-variance. Binary
"grounded vs. needs-review" accuracy for the NLI systems is **0.62** (vs. the 0.54 majority),
and the flag recall of **0.80** means the tool surfaces four of every five claims a reviewer
would want to check.

Per-class for zero-shot NLI: supported P0.74 / R0.47 / F0.58; not_mentioned P0.51 /
R0.53 / F0.52; contradicted R1.00 but P0.15 (only 3 gold instances). The low
supported-recall and low contradicted-precision are the two error modes analysed in §6.

### 5.2 Ablation — the NER contradiction gate

Off-topic notes containing negative words produce **false contradictions**. Gating by
shared topical entity removes them, but a naive *hard* gate over-corrects:

| configuration | false contradictions | true-contra kept | macro-F1 |
|---|---|---|---|
| no gate (raw NLI) | 17 | 3 / 3 | 0.452 |
| hard gate (`--gate_ner`) | 7 | 1 / 3 | 0.440 |
| **soft gate (`--gate_keep_sim 0.55`)** | 13 | **3 / 3** | **0.478** |

The hard gate blocks 10 false contradictions but, due to lexicon coverage gaps, also
wrongly blocks 2 of the 3 true contradictions. The **soft gate** keeps an off-topic
contradiction if its retrieval similarity is nonetheless high (≥ 0.55), recovering all
true contradictions while still removing 4 false ones — giving the best macro-F1 of any
configuration. This is the chosen final system. (The keep-similarity threshold was
selected on the labelled set; given only 3 gold contradictions the margins are small,
so the gain over raw NLI should be read as *modest but consistent*.)

### 5.3 Negative result — confidence is not calibrated (entropy rejected)

I tested whether Shannon-entropy-based **selective prediction** could improve results
by abstaining on low-confidence claims. The risk-coverage curve is **inverted**: on the
most-confident 25% of predictions accuracy is **0.409**, *below* the 0.511 over all
claims. The reason is diagnostic: the model's high-confidence predictions include its
confidently-wrong false contradictions (contradiction ≈ 0.9–1.0). Entropy would
therefore tell us to trust exactly the wrong predictions, so I **do not** adopt it —
this is reported as a calibration finding, and it independently motivates the entity
gate.

## 6. Error analysis and discussion

The 44 errors of the raw NLI system fall into two dominant buckets:

- **False contradictions (17).** Off-topic notes with negative vocabulary are scored as
  contradictions (e.g. a claim about "impulse control" flagged against an unrelated note
  about an aptitude assessment). The NER soft-gate targets this bucket (§5.2).
- **Missed support (19).** A supported claim is labelled *not_mentioned* because it is an
  **abstractive summary** — it adds specifics (a date such as *"im Juli 2024"*) or bundles
  several facts into one sentence — that **no single note entails**, even when the notes
  clearly support it collectively. Stripping the date raised entailment from 0.02 to 0.53
  on one representative claim, but rule-based claim decomposition did not yield a
  consistent overall gain.

This second bucket is the honest core finding: **abstractive report claims versus
granular case notes create a granularity mismatch**, which sentence-level NLI cannot
fully bridge. This is precisely the limitation documented for MiniCheck (Tang et al.,
2024), whose authors show sentence-level entailment models fail on claims requiring
synthesis across multiple sentences. Consequently, system **accuracy sits at roughly the
majority baseline** — not because the model is uninformative (its macro-F1 and flag
recall are well above baseline and the lexical system) but because the imbalance and the
missed-support bucket cap raw accuracy. I report this openly rather than selecting a
flattering metric.

## 7. NLP techniques used (course mapping)

The capstone demonstrates the following techniques from the course (the minimum was
four):

1. **Word/Sentence Embeddings** — multilingual SBERT retrieval (`faithcheck.py`).
2. **Transfer Learning** — zero-shot transfer of an XNLI-pretrained model to the LuV
   domain, no fine-tuning (`faithcheck.py`).
3. **Language Models** — the mDeBERTa NLI transformer and MiniLM encoder as the backbone.
4. **Evaluation Methods** — accuracy/macro-F1/per-class/confusion, flag precision-recall,
   majority baseline, dev/test threshold tuning (`evaluate.py`, `tune.py`).
5. **Named Entity Recognition** — offline domain NER entity-gate (`ner.py`).

Sub-word tokenization (inside the transformer tokenizers) and the attention mechanism
(inside the models) are additionally present, though not as standalone contributions.

## 8. Related work

- **SummaC** (Laban et al., 2022) — NLI-based inconsistency detection with sentence-level
  aggregation; the methodological basis for our aggregation step.
- **MiniCheck** (Tang et al., 2024) — retrieve-then-check grounded fact-checking; documents
  the multi-sentence limitation I observe.
- **FEVER** (Thorne et al., 2018) — the supported/refuted/not-enough-info claim-verification
  task our labels mirror.
- **AlignScore** (Zha et al., 2023) — unified alignment metric for factual consistency.
- **NLI4CT** (Jullien et al., SemEval-2024) and clinical contradiction detection — the closest
  published task (entailment/contradiction of statements against real clinical reports),
  which shares our numerical/temporal-reasoning difficulty.
- **Annotation artifacts / hypothesis-only bias** (Poliak et al., 2018; Gururangan et al.,
  2018) — the phenomenon behind our calibration finding (§5.3) and the previous project's
  negation-shortcut result.

## 9. Limitations and future work

- The gold set is small (90 claims, 3 contradictions); differences of a few points are within
  noise, and the keep-similarity threshold is tuned on this set. In particular, most of the
  macro-F1 gap over the lexical baseline comes from the 3-instance contradicted class, so the
  robust evidence of improvement is flag recall and not_mentioned F1, not macro-F1 alone.
- **Single annotator, no inter-annotator agreement.** The 90 gold labels reflect one person's
  judgment; abstractive-summary claims are often genuinely borderline, and a second annotator
  would let us report agreement and quantify label noise — the first thing a larger study should add.
- The domain NER is lexicon-based, so its topical coverage is incomplete.
- The largest remaining error bucket (missed support) needs claim decomposition or a
  fact-checking model trained for multi-sentence synthesis (e.g. MiniCheck), ideally in
  German; naive rule-based decomposition was inconclusive here.
- A larger, natively German gold set and per-competence-area analysis are the natural next steps.

## 10. Conclusion

Applying the SummaC/MiniCheck grounded-fact-checking recipe — retrieval + zero-shot
multilingual NLI + entity-gated aggregation — to German LuV reports yields a **local,
privacy-preserving faithfulness checker** that produces cited, confidence-scored verdicts and
**improves on the previous lexical baseline where it matters most: flag recall 0.80 vs 0.54**
(macro-F1 0.48 vs 0.33). The NER soft-gate adds a further small refinement concentrated in the
contradiction class; resting on only three gold contradictions, it should be read as
suggestive rather than conclusive. An honest evaluation on human labels shows the system's
value lies in **flagging** rather than raw accuracy, which is capped by class imbalance and a
genuine abstractive-claim / granular-note **granularity mismatch** — a finding consistent
with the published limits of sentence-level NLI.

---

### References

*(Verify formatting against your course's citation style.)*

- Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022). *SummaC: Re-Visiting
  NLI-based Models for Inconsistency Detection in Summarization.* TACL. arXiv:2111.09525.
- Tang, L., Laban, P., & Durrett, G. (2024). *MiniCheck: Efficient Fact-Checking of LLMs on
  Grounding Documents.* EMNLP. arXiv:2404.10774.
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). *FEVER: a Large-scale
  Dataset for Fact Extraction and VERification.* NAACL.
- Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). *AlignScore: Evaluating Factual Consistency with
  a Unified Alignment Function.* ACL. arXiv:2305.16739.
- Jullien, M., et al. (2024). *SemEval-2024 Task 2: Safe Biomedical Natural Language Inference
  for Clinical Trials (NLI4CT).*
- Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. (2018). *Hypothesis
  Only Baselines in Natural Language Inference.* *SEM.
- Gururangan, S., et al. (2018). *Annotation Artifacts in Natural Language Inference Data.* NAACL.
- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2022). *Less Annotating, More
  Classifying* (mDeBERTa-v3 XNLI model).
