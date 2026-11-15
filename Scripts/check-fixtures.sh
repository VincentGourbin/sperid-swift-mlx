#!/usr/bin/env bash
# Régénère les fixtures tiny dans un dossier temporaire et les compare au bit à parity/.
# Une différence = fixture éditée à la main, générateur modifié sans régénération, ou amont changé.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -x .venv-ref/bin/python ] || { echo "FIXTURES MISSING .venv-ref (Scripts/reference/setup_venv.sh)"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
.venv-ref/bin/python -I Scripts/reference/tiny_fixtures.py "$tmp" > /dev/null || { echo "FIXTURES FAILED (générateur)"; exit 1; }
.venv-ref/bin/python -I Scripts/reference/birefnet_fixtures.py --tiny-only --tiny-out "$tmp" > /dev/null || { echo "FIXTURES FAILED (BiRefNet)"; exit 1; }
status=0
for f in "$tmp"/*.safetensors; do
  name=$(basename "$f")
  if [ ! -f "parity/$name" ]; then echo "FIXTURES MISSING parity/$name"; status=1; continue; fi
  # The `generated` date is the only field allowed to differ: compare tensors and the other metadata.
  .venv-ref/bin/python -I - "$f" "parity/$name" <<'PY' || status=1
import sys
from safetensors import safe_open
a, b = (safe_open(p, "pt") for p in sys.argv[1:3])
ma, mb = ({k: v for k, v in (x.metadata() or {}).items() if k != "generated"} for x in (a, b))
assert ma == mb, f"metadata differs: {ma} vs {mb}"
assert set(a.keys()) == set(b.keys()), "tensor names differ"
bad = [k for k in a.keys() if not a.get_tensor(k).equal(b.get_tensor(k))]
assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:3]}"
print(f"{sys.argv[2]}: {len(list(a.keys()))} tensors identical, upstream {mb.get('upstream_sha', '?')[:7]}")
PY
done
[ "$status" -eq 0 ] && echo "FIXTURES OK" || { echo "FIXTURES FAILED"; exit 1; }
