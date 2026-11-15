#!/usr/bin/env bash
# Concatène plan/*.md dans l'ordre (lecture humaine ; interdit à l'exécutant, cf. AGENTS.md).
cd "$(dirname "$0")/.."
for f in plan/*.md; do cat "$f"; echo; done
