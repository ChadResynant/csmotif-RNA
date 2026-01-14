#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"
SCAN_RESULT=$(rg -n 'shell\s*=\s*True|\\bos\\.system\\s*\\(|\\bos\\.popen\\s*\\(' . --type py --glob '!vendor/**' --glob '!tools/ci/**' --glob '!tools/hardening/**' --glob '!.git/**' --glob '!archive/**' --glob '!__pycache__/**' --glob '!*.pyc' 2>/dev/null || true)
if [[ -n "$SCAN_RESULT" ]]; then
  echo "❌ Shell hazards detected (hard-mode gate failed):"
  echo ""
  echo "$SCAN_RESULT"
  echo ""
  exit 1
fi
echo "✓ Hard-mode shell hygiene: 0 hazards detected"
exit 0
