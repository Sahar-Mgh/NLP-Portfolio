#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Runs the full pipeline end-to-end and prints the evaluation table.
#
# DEFAULT: runs on the bundled SYNTHETIC (mock) data that ships with the repo.
#   This is a smoke test — it proves the pipeline works and produces all three
#   verdicts. The numbers are NOT the REPORT.md numbers (mock data is tiny).
#
# REAL DATA: the results in REPORT.md come from real institute data that is NOT
#   in this repo (privacy — see README). If you hold that data locally under
#   data/*.jsonl, reproduce the report with:
#         REAL=1 bash run.sh
#
# Usage:  bash run.sh          # mock demo
#         REAL=1 bash run.sh   # reproduce report (needs local real data)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

if [[ "${REAL:-0}" == "1" ]]; then
  CLAIMS=data/claims.jsonl;      NOTES=data/notes.jsonl;      LABELS=data/labels.jsonl;      S=""
  echo ">>> REAL institute data (results are private — do not commit results/*.jsonl)"
else
  CLAIMS=data/claims.mock.jsonl; NOTES=data/notes.mock.jsonl; LABELS=data/labels.mock.jsonl; S=".mock"
  echo ">>> SYNTHETIC mock data (demo only; numbers will not match REPORT.md)"
fi

echo "[1/4] lexical baseline"
python faithcheck.py --method lexical --claims "$CLAIMS" --notes "$NOTES" --out "results/preds_lexical${S}.jsonl"
echo "[2/4] zero-shot NLI (downloads models on first run; minutes on CPU)"
python faithcheck.py --method nli     --claims "$CLAIMS" --notes "$NOTES" --out "results/preds_nli${S}.jsonl"
echo "[3/4] NLI + soft NER gate (final system)"
python faithcheck.py --method nli --gate_ner --gate_keep_sim 0.55 --claims "$CLAIMS" --notes "$NOTES" --out "results/preds_nli_softgate${S}.jsonl"
echo "[4/4] evaluate against gold labels"
python evaluate.py \
  --preds "results/preds_lexical${S}.jsonl" "results/preds_nli${S}.jsonl" "results/preds_nli_softgate${S}.jsonl" \
  --names old-lexical zero-shot-NLI NLI-softgate --labels "$LABELS"
