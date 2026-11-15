---
okf_version: "0.1"
---

# Base de connaissances — sperid-swift-mlx

Journal d'ingénierie du portage (bundle OKF). `log.md` porte l'historique horodaté ; les
sous-répertoires accumulent les conclusions durables.

## Benchmarks
Mesures reproductibles (état des horloges GPU, écart ±2 %). Rien pour l'instant : coûts attendus dans `plan/03-budget.md`.

## Decisions
- [Source des poids des encodeurs](decisions/weights-source.md) — DINOv3 gated sans accès au 2026-11-15 ; dépôt officiel recommandé, repack Comfy en repli ; FLUX.2-dev accessible, encodeur aussi présent en local.
- [Organisation du paquet](decisions/package-layout.md) — `SperidCore` partagé + `IdolCore` + `idol` ; copie depuis les dépôts frères, pas de dépendance ; octree sur CPU ; bruit injectable.
- [Profils de référence (prévus)](decisions/reference-profiles.md) — matrice `<bits>bit-fast|lean` + famille `turbo/`, tout « à mesurer ».

## Pitfalls
Numérotés comme `plan/11-pieges.md`.
- [`Float.pi` arrondi vers zéro](pitfalls/swift-float-pi-rounded-toward-zero.md) — un ulp sous le π de torch, 2,4e-4 sur sin/cos à 3000 rad ; `Float(Double.pi)` ; l'`exp` de torch à ~1 ulp impose une tolérance trig de 3e-4 (n° 20).

## Investigations
Rien pour l'instant.

## Transmissions et mémoire
- [Mémoire d'agent](agent-memory.md) — conventions de Vincent, état du projet : à lire en début de session.
