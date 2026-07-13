"""
faithcheck.py -- note-grounded faithfulness checker for LuV report claims.
=========================================================================
For each report CLAIM, retrieve the most relevant NOTES for that student, score
each (note, claim) pair, and aggregate SummaC-style into a verdict, with a
CITATION and a CONFIDENCE score:

    claim ──(retrieve top-k notes)──▶ score(note, claim) per note
          ──(max-aggregate)────────▶ supported | contradicted | not_mentioned
                                       (+ binary: grounded | needs_review)
                                       + citation (the note that drove it) + confidence

Scorers (interchangeable, plug into the SAME retrieve->aggregate harness):
  --method nli      zero-shot multilingual NLI (mDeBERTa-xnli)   [reads the evidence]
  --method lexical  the previous project's TF-IDF + negation-cue heuristic  [old baseline]

Improvements (diagnosed on the real LuV data -- see README):
  --decompose       split each claim into atomic sub-claims and STRIP DATES, then
                    require every sub-claim to hold. Fixes "missed support" where a
                    claim adds specifics (e.g. 'im Juli 2024') that no single note states.
  --gate_sim FLOAT  only count a note's CONTRADICTION if its retrieval similarity clears
                    this floor -> suppresses false contradictions from off-topic notes.

No training -> no negation shortcut. Runs locally on CPU. Note embeddings are
precomputed once per student so it scales to thousands of note-sentences.

Run:
  python faithcheck.py --method nli --claims data/claims.jsonl --notes data/notes.jsonl --out results/preds_nli.jsonl
  python faithcheck.py --method nli --decompose --gate_sim 0.5 --claims data/claims.jsonl --notes data/notes.jsonl --out results/preds_nli_improved.jsonl
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ner import share_topic

try:                                    # make Windows console UTF-8 safe
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                       encoding="utf-8")


# --------------------------------------------------------------------------- #
# Claim decomposition: strip dates, split compound sentences into atomic claims.
# --------------------------------------------------------------------------- #
_MONTHS = ("Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|"
           "November|Dezember")
_DATE_RX = re.compile(r"\b(?:im|am|seit|ab|bis|zum|vom|Anfang|Mitte|Ende)?\s*"
                      r"\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_MONTHYEAR_RX = re.compile(rf"\b(?:im|seit|ab|bis|Anfang|Mitte|Ende)?\s*(?:{_MONTHS})"
                           rf"(?:\s+\d{{4}})?\b", re.IGNORECASE)
_YEAR_RX = re.compile(r"\b(?:im Jahr(?:e)?\s+)?\d{4}\b")


def strip_dates(s):
    for rx in (_DATE_RX, _MONTHYEAR_RX, _YEAR_RX):
        s = rx.sub(" ", s)
    return re.sub(r"\s{2,}", " ", s).strip(" ,;.")


def decompose_claim(text):
    """Date-stripped claim, split into atomic sub-claims ONLY when the split yields
    >=2 well-formed parts (each >=4 words). Otherwise keep the whole date-stripped
    claim -- avoids the malformed fragments ('...benötigt er') that naive splitting
    produces and that then tank the support score."""
    s = strip_dates(text)
    parts = re.split(r"\s*;\s*|,?\s+(?:jedoch|aber|allerdings|während|wohingegen)\s+", s)
    frags = []
    for p in parts:
        subj = r"(?:er|sie|es|Herr|Frau|der TN|die TN|Herrn)"
        frags += re.split(rf"\s+und\s+(?={subj}\b)", p)
    frags = [f.strip(" ,;.") for f in frags if f.strip()]
    if len(frags) >= 2 and all(len(f.split()) >= 4 for f in frags):
        return frags
    return [s if s.strip() else text]


# --------------------------------------------------------------------------- #
# Retrieval: multilingual SBERT if available, TF-IDF char n-gram fallback.
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMB_MODEL)
            self.mode = f"sbert:{EMB_MODEL.split('/')[-1]}"
            self.sbert = True
        except Exception as exc:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self._Tfidf, self._norm = TfidfVectorizer, normalize
            self.mode = "tfidf:char_wb(3,5)"
            self.sbert = False
            banner = (
                "\n" + "!" * 74 +
                f"\n!! WARNING: sentence-transformers unavailable ({exc.__class__.__name__})."
                "\n!! Falling back to a TF-IDF retriever. This DOES NOT reproduce the"
                "\n!! numbers in REPORT.md -- the reported results require SBERT retrieval."
                "\n!! Fix:  pip install -r requirements.txt   (then re-run)\n" +
                "!" * 74 + "\n")
            print(banner, file=sys.stderr)

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=64))

    def rank(self, query, docs, doc_emb=None):
        if not docs:
            return [], np.array([])
        if self.sbert:
            q = np.asarray(self.model.encode([query], normalize_embeddings=True))[0]
            D = doc_emb if doc_emb is not None else self.encode(docs)
            sims = D @ q
        else:
            vec = self._Tfidf(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit([query] + docs)
            X = self._norm(vec.transform([query] + docs)).toarray()
            sims = X[1:] @ X[0]
        return list(np.argsort(-sims)), sims


# --------------------------------------------------------------------------- #
# Scorers (same interface: .score(premise, hypothesis) -> ent/neutral/contra)
# --------------------------------------------------------------------------- #
class NLIScorer:
    name = "nli"

    def __init__(self, model_name=NLI_MODEL):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.idmap = {i: self.model.config.id2label[i].lower() for i in self.model.config.id2label}
        self.desc = f"zero-shot NLI ({model_name})"

    def score(self, premise, hypothesis):
        with self.torch.no_grad():
            x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
            p = self.torch.softmax(self.model(**x).logits[0], dim=-1).tolist()
        out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for i, v in enumerate(p):
            out[self.idmap[i]] = float(v)
        return out


class LexicalScorer:
    name = "lexical"
    SIM_LOW = 0.06
    NEG_WORDS = re.compile(r"\b(nicht|kein(?:e|en|er)?|nie|niemals|ohne)\b", re.IGNORECASE)
    NEG_PREFIX = re.compile(r"\bun(motiviert|zuverlässig|pünktlich|selbstständig|konzentriert|"
                            r"freundlich|sicher|geeignet|regelmäßig|auffällig)", re.IGNORECASE)

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self._Tfidf, self._norm = TfidfVectorizer, normalize
        self.desc = "lexical TF-IDF + negation-cue heuristic (previous project's baseline)"

    def _neg(self, text):
        t = text or ""
        return len(self.NEG_WORDS.findall(t)) + len(self.NEG_PREFIX.findall(t))

    def score(self, premise, hypothesis):
        vec = self._Tfidf(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit([premise, hypothesis])
        P = self._norm(vec.transform([premise]))
        H = self._norm(vec.transform([hypothesis]))
        sim = float((P @ H.T).toarray()[0, 0])
        dneg = self._neg(hypothesis) - self._neg(premise)
        rel = (sim - self.SIM_LOW) * 12.0
        logits = np.array([rel - 1.5 * max(dneg, 0), rel + 2.2 * max(dneg, 0) - 0.5, -rel + 0.5])
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return {"entailment": float(p[0]), "contradiction": float(p[1]), "neutral": float(p[2])}


def make_scorer(method):
    return NLIScorer() if method == "nli" else LexicalScorer()


# --------------------------------------------------------------------------- #
# Per-claim decision: (optionally decompose) -> retrieve -> score -> aggregate.
# With one sub-claim and gate_sim=0 this reproduces plain per-note max-aggregation.
# --------------------------------------------------------------------------- #
def judge_claim(claim_row, notes, retr, scorer, k, tau_sup, tau_con, doc_emb=None,
                decompose=False, gate_sim=0.0, gate_ner=False, gate_keep_sim=2.0):
    texts = [n["text"] for n in notes]

    def cite(idx):
        if idx is None:
            return None
        n = notes[idx]
        return {"note_id": n.get("note_id"), "date": n.get("date"), "author": n.get("author"),
                "title": n.get("title"), "text": n.get("text", "")[:240]}

    if not texts:
        return dict(verdict="not_mentioned", binary="needs_review", confidence=0.0,
                    support_score=0.0, contradiction_score=0.0, citation=None,
                    subclaims=[], evidence=[])

    subs = decompose_claim(claim_row["claim"]) if decompose else [claim_row["claim"]]
    sub_out = []
    for sc in subs:
        order, sims = retr.rank(sc, texts, doc_emb=doc_emb)
        topk = order[:k]
        best_sup, sup_idx = -1.0, topk[0]
        best_con, con_idx = -1.0, None
        ev = []
        for idx in topk:
            s = scorer.score(texts[idx], sc)
            ev.append({"note_id": notes[idx].get("note_id"), "sim": round(float(sims[idx]), 3),
                       "entailment": round(s["entailment"], 3),
                       "contradiction": round(s["contradiction"], 3),
                       "neutral": round(s["neutral"], 3)})
            if s["entailment"] > best_sup:
                best_sup, sup_idx = s["entailment"], idx
            on_topic = ((not gate_ner) or share_topic(sc, texts[idx])
                        or float(sims[idx]) >= gate_keep_sim)   # softer gate: keep off-topic if very on-point
            if float(sims[idx]) >= gate_sim and on_topic and s["contradiction"] > best_con:
                best_con, con_idx = s["contradiction"], idx
        if best_con < 0:
            best_con = 0.0
        sub_out.append({"subclaim": sc, "support": round(best_sup, 3), "sup_idx": sup_idx,
                        "contradiction": round(best_con, 3), "con_idx": con_idx, "evidence": ev})

    sup_agg = min(s["support"] for s in sub_out)        # every sub-claim must be backed
    weak = min(sub_out, key=lambda s: s["support"])     # weakest-supported sub-claim
    con_lead = max(sub_out, key=lambda s: s["contradiction"])
    con_agg = con_lead["contradiction"]

    if con_agg >= tau_con and con_agg >= sup_agg:
        verdict, conf, c_idx, drive = "contradicted", con_agg, con_lead["con_idx"], con_lead
    elif sup_agg >= tau_sup:
        verdict, conf, c_idx, drive = "supported", sup_agg, weak["sup_idx"], weak
    else:
        verdict, conf, c_idx, drive = "not_mentioned", round(1 - max(sup_agg, con_agg), 3), \
                                      sub_out[0]["sup_idx"], sub_out[0]

    return dict(verdict=verdict,
                binary=("grounded" if verdict == "supported" else "needs_review"),
                confidence=round(float(conf), 3),
                support_score=round(float(sup_agg), 3),
                contradiction_score=round(float(con_agg), 3),
                citation=cite(c_idx),
                subclaims=[{"subclaim": s["subclaim"], "support": s["support"],
                            "contradiction": s["contradiction"]} for s in sub_out],
                evidence=drive["evidence"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--method", choices=["nli", "lexical"], default="nli")
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=8, help="how many notes to retrieve per (sub)claim")
    ap.add_argument("--tau_sup", type=float, default=0.55, help="entailment threshold -> supported")
    ap.add_argument("--tau_con", type=float, default=0.55, help="contradiction threshold -> flag")
    ap.add_argument("--decompose", action="store_true", help="split claims into atomic sub-claims + strip dates")
    ap.add_argument("--gate_sim", type=float, default=0.0, help="min retrieval sim for a contradiction to count")
    ap.add_argument("--gate_ner", action="store_true", help="only count a contradiction if the note shares a SKILL/AREA entity with the claim (domain NER gate)")
    ap.add_argument("--gate_keep_sim", type=float, default=2.0, help="with --gate_ner, KEEP an off-topic contradiction anyway if its retrieval sim >= this (softer gate; e.g. 0.55)")
    ap.add_argument("--limit", type=int, default=0, help="only judge the first N claims (0 = all)")
    args = ap.parse_args()
    if args.out is None:
        tag = args.method + ("_improved" if (args.decompose or args.gate_sim) else "")
        args.out = f"results/preds_{tag}.jsonl"

    claims = read_jsonl(args.claims)
    if args.limit:
        claims = claims[:args.limit]
    notes = read_jsonl(args.notes)
    notes_by_student = defaultdict(list)
    for n in notes:
        notes_by_student[n["student_id"]].append(n)

    print(f"loading retriever + '{args.method}' scorer...")
    retr, scorer = Retriever(), make_scorer(args.method)
    print(f"retriever = {retr.mode}\nscorer    = {scorer.desc}")
    print(f"claims={len(claims)}  notes={len(notes)}  students={len(notes_by_student)}  k={args.k}  "
          f"tau_sup={args.tau_sup}  tau_con={args.tau_con}  decompose={args.decompose}  gate_sim={args.gate_sim}")

    index = {}
    if retr.sbert:
        for sid, ns in notes_by_student.items():
            index[sid] = retr.encode([n["text"] for n in ns])
        print(f"precomputed note embeddings for {len(index)} students\n")

    verbose = len(claims) <= 30
    out_rows, dist = [], Counter()
    for i, c in enumerate(claims, 1):
        sid = c["student_id"]
        res = judge_claim(c, notes_by_student.get(sid, []), retr, scorer, args.k,
                          args.tau_sup, args.tau_con, doc_emb=index.get(sid),
                          decompose=args.decompose, gate_sim=args.gate_sim, gate_ner=args.gate_ner,
                          gate_keep_sim=args.gate_keep_sim)
        row = {**{kk: c.get(kk) for kk in ("claim_id", "student_id", "area", "claim")},
               "method": args.method, **res}
        out_rows.append(row)
        dist[res["verdict"]] += 1
        if verbose:
            cite = res["citation"]["note_id"] if res["citation"] else "-"
            print(f"  {c['claim_id']:26s} {res['verdict']:13s} conf={res['confidence']:.3f} "
                  f"(sup={res['support_score']:.2f} con={res['contradiction_score']:.2f}) cite={cite}")
        elif i % 25 == 0 or i == len(claims):
            print(f"  [{i:4d}/{len(claims)}]  {dict(dist)}")

    write_jsonl(args.out, out_rows)
    print(f"\nsaved -> {args.out}  ({len(out_rows)} claims)")
    print("verdict distribution:", dict(dist))


if __name__ == "__main__":
    main()
