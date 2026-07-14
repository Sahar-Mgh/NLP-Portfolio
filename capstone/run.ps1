# Run the full pipeline and print the evaluation table.
#
# By default this runs on the bundled mock data as a smoke test: it exercises
# every method, but the numbers won't match REPORT.md (the mock set is tiny).
#
# The REPORT.md numbers come from real data that isn't committed for privacy
# reasons (see README). With that data under data/*.jsonl, set REAL=1 to
# reproduce the report.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run.ps1              # mock demo
#         $env:REAL=1; powershell -ExecutionPolicy Bypass -File run.ps1   # reproduce report
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force -Path results | Out-Null

if ($env:REAL -eq "1") {
    $CLAIMS = "data/claims.jsonl"; $NOTES = "data/notes.jsonl"; $LABELS = "data/labels.jsonl"; $S = ""
    Write-Host ">>> REAL data (results are private; do not commit results/*.jsonl)"
} else {
    $CLAIMS = "data/claims.mock.jsonl"; $NOTES = "data/notes.mock.jsonl"; $LABELS = "data/labels.mock.jsonl"; $S = ".mock"
    Write-Host ">>> SYNTHETIC mock data (demo only; numbers will not match REPORT.md)"
}

function Step($label, [scriptblock]$cmd) {
    Write-Host $label; & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Error "Step failed: $label"; exit 1 }
}

Step "[1/4] lexical baseline" {
    python faithcheck.py --method lexical --claims $CLAIMS --notes $NOTES --out "results/preds_lexical$S.jsonl"
}
Step "[2/4] zero-shot NLI (downloads models on first run; minutes on CPU)" {
    python faithcheck.py --method nli --claims $CLAIMS --notes $NOTES --out "results/preds_nli$S.jsonl"
}
Step "[3/4] NLI + soft NER gate (final system)" {
    python faithcheck.py --method nli --gate_ner --gate_keep_sim 0.55 --claims $CLAIMS --notes $NOTES --out "results/preds_nli_softgate$S.jsonl"
}
Step "[4/4] evaluate against gold labels" {
    python evaluate.py --preds "results/preds_lexical$S.jsonl" "results/preds_nli$S.jsonl" "results/preds_nli_softgate$S.jsonl" `
        --names old-lexical zero-shot-NLI NLI-softgate --labels $LABELS
}
