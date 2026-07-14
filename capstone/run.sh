#!/usr/bin/env bash
# Run the full pipeline and print the evaluation table.
#
# By default this runs on the bundled mock data as a smoke test: it exercises
# every method, but the numbers won't match REPORT.md (the mock set is tiny).
#
# The REPORT.md numbers come from real data that isn't committed for privacy
# reasons (see README). With that data under data/*.jsonl, set REAL=1 to
# reproduce the report.
#
# Usage:
#   bash run.sh          # mock demo
#   REAL=1 bash run.sh   # reproduce report (needs local real data)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

if [[ "${REAL:-0}" == "1" ]]; then
  CLAIMS=data/claims.jsonl;      NOTES=data/notes.jsonl;      LABELS=data/labels.jsonl;      S=""
  echo ">>> REAL data (results are private; do not commit results/*.jsonl)"
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
