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
# decode: the Swift PLY writer on the reference cloud must reproduce `save_ply` (Python side)
if [ "$comp" = "decode" ] && [ "$status" -eq 0 ]; then
  root="${SPERID_MODELS_DIR:-$HOME/Library/Caches/models}"
  if .venv-ref/bin/python -I Scripts/reference/compare_ply.py .local-runs/parity/decode_reference_cloud.ply \
       "$root/parity/uncond.ply" --tol 1e-5 >> ".local-runs/parity-$comp.log" 2>&1; then
    echo "PARITY OK decode.ply $(grep '^PLY' ".local-runs/parity-$comp.log" | tail -1)" >> ".local-runs/parity-$comp.log"
  else
    echo "PARITY FAILED decode.ply $(grep '^PLY' ".local-runs/parity-$comp.log" | tail -1)" >> ".local-runs/parity-$comp.log"
    status=1
  fi
fi
grep "^PARITY" ".local-runs/parity-$comp.log" | head -40
last=$(grep "^PARITY OK $comp \|^PARITY FAILED" ".local-runs/parity-$comp.log" | tail -1)
[ "$status" -eq 0 ] && [[ "$last" == PARITY\ OK* ]] && exit 0
echo "${last:-PARITY FAILED (statut $status)} — journal : .local-runs/parity-$comp.log"
exit 1
