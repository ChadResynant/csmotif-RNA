#!/usr/bin/env bash
# Shell Hygiene Gate (Baseline-Diff Mode)
#
# Two strategies available:
#   1. BASELINE-DIFF (this script):  Fails if count increases from baseline
#      Use for: Legacy scanning, incremental burn-down
#      Command: ./shell_hygiene_gate.sh [--update-baseline]
#
#   2. HARD-MODE (recommended for CI): Zero-tolerance, rejects ANY hazard
#      Use for: CI/CD, src/python/ only (active code)
#      Command: ./shell_hygiene_gate_hardmode.sh
#
# NOTE: Since baseline is now 0, hard-mode is the preferred CI strategy.
#       See shell_hygiene_gate_hardmode.sh for production CI runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
BASELINE="$REPO_ROOT/tools/ci/shell_hazards_baseline.txt"
CURRENT="/tmp/shell_hazards_current.txt"

cd "$REPO_ROOT"

# Generate current snapshot (run from repo root, scan src/ only)
echo "Scanning for shell hazards..."

grep -r "shell\s*=\s*True\|os\.system\s*(\|os\.popen\s*(" src/ \
  --include="*.py" \
  2>/dev/null \
  | grep -v ".pyc" \
  | sort > "$CURRENT" || true

BASELINE_COUNT=$(wc -l < "$BASELINE" 2>/dev/null || echo 0)
CURRENT_COUNT=$(wc -l < "$CURRENT" 2>/dev/null || echo 0)

echo "Baseline: $BASELINE_COUNT known hazards"
echo "Current:  $CURRENT_COUNT hazards found"

# Handle --update-baseline flag
if [[ "${1:-}" == "--update-baseline" ]]; then
    echo ""
    echo "Updating baseline (you fixed hazards!)..."
    cp "$CURRENT" "$BASELINE"
    echo "✓ Baseline updated: $BASELINE_COUNT → $CURRENT_COUNT hazards"
    exit 0
fi

# Compare: fail only if count increased
if [[ $CURRENT_COUNT -gt $BASELINE_COUNT ]]; then
    echo ""
    echo "❌ ERROR: Shell hazard count increased!"
    echo ""
    echo "Baseline ($BASELINE_COUNT):"
    head -5 "$BASELINE"
    echo ""
    echo "Current ($CURRENT_COUNT):"
    head -5 "$CURRENT"
    echo ""
    echo "NEW HAZARDS DETECTED:"
    diff "$BASELINE" "$CURRENT" | grep "^>" | head -5
    echo ""
    echo "NEXT STEPS:"
    echo "1. Fix the new hazards (convert shell=True to run_argv)"
    echo "2. Re-run this gate: ./tools/ci/shell_hygiene_gate.sh"
    echo ""
    echo "If you REMOVED hazards and baseline is stale:"
    echo "   ./tools/ci/shell_hygiene_gate.sh --update-baseline"
    exit 1
elif [[ $CURRENT_COUNT -lt $BASELINE_COUNT ]]; then
    echo "✓ Good: Hazard count decreased ($BASELINE_COUNT → $CURRENT_COUNT)"
    echo "  Update baseline to reflect progress:"
    echo "  ./tools/ci/shell_hygiene_gate.sh --update-baseline"
    exit 0
else
    echo "✓ OK: Hazard count unchanged"
    exit 0
fi
