#!/usr/bin/env bash
# Parité tier 2 : Scripts/check-parity.sh <composant>. Exige SPERID_MODELS_DIR (ou le défaut) et le
# binaire Release (Scripts/check-build.sh d'abord). Journal : .local-runs/parity-<c>.log
set -uo pipefail
cd "$(dirname "$0")/.."
comp="${1:?composant requis (ex. decode)}"
mkdir -p .local-runs
bin=.xcodebuild/Build/Products/Release/idol
[ -x "$bin" ] || { echo "PARITY FAILED : binaire absent, lancer Scripts/check-build.sh"; exit 1; }
"$bin" parity "$comp" ${SPERID_MODELS_DIR:+--models-dir "$SPERID_MODELS_DIR"} > ".local-runs/parity-$comp.log" 2>&1
status=$?
grep "^PARITY" ".local-runs/parity-$comp.log" | head -40
last=$(grep "^PARITY OK\|^PARITY FAILED" ".local-runs/parity-$comp.log" | tail -1)
[ "$status" -eq 0 ] && [[ "$last" == PARITY\ OK* ]] && exit 0
echo "${last:-PARITY FAILED (statut $status)} — journal : .local-runs/parity-$comp.log"
exit 1
