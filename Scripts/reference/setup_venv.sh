#!/usr/bin/env bash
# Crée .venv-ref (gitignoré) : le SDK idol amont en editable + safetensors/numpy pour les fixtures.
# Prérequis : Scripts/link-references.sh (reference/idol-sdk). Python 3.10+ (3.12 par défaut, via uv).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -d reference/idol-sdk ] || { echo "reference/idol-sdk absent : lancer Scripts/link-references.sh"; exit 1; }
PY="${SPERID_REF_PYTHON:-3.12}"
if command -v uv >/dev/null; then
  [ -x .venv-ref/bin/python ] || uv venv --quiet --python "$PY" .venv-ref
  uv pip install --quiet --python .venv-ref/bin/python -e reference/idol-sdk safetensors numpy
else
  [ -x .venv-ref/bin/python ] || "python$PY" -m venv .venv-ref
  .venv-ref/bin/pip install --quiet -e reference/idol-sdk safetensors numpy
fi
.venv-ref/bin/python -I -c "import idol, torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
