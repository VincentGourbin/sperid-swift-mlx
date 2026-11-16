---
okf_version: "0.1"
---

# Base de connaissances — sperid-swift-mlx

Journal d'ingénierie du portage (bundle OKF). `log.md` porte l'historique horodaté ; les
sous-répertoires accumulent les conclusions durables.

## Benchmarks
Mesures reproductibles (état des horloges GPU, écart ±2 %).
- [Profil M3 Max du pipeline Swift](benchmarks/m3max-profile-2026-11-15.md) — turbo 22 s (débruitage 77 %, GPU 100 %, aucun goulot CPU) ; bloc joint : SDPA 59 % ; fusion rotation+RMSNorm −4 % exacte ; cache 2 Go : pic process 34 → 11 Go ; pic MLX 11 Go = convolution pleine résolution du VAE (im2col).

## Decisions
- [Source des poids des encodeurs](decisions/weights-source.md) — DINOv3 gated sans accès au 2026-11-15 ; dépôt officiel recommandé, repack Comfy en repli ; FLUX.2-dev accessible, encodeur aussi présent en local.
- [Organisation du paquet](decisions/package-layout.md) — `SperidCore` partagé + `IdolCore` + `idol` ; copie depuis les dépôts frères, pas de dépendance ; octree sur CPU ; bruit injectable.
- [Formats de sortie](decisions/output-formats.md) — PLY fait (Blender 5.3 l'importe) ; RealityKit 27 rend les splats nativement (vérifié SDK) ; USD 26.03 splats sans Quick Look ; USDZ universel = maillage à extraire.
- [Profils de référence (prévus)](decisions/reference-profiles.md) — matrice `<bits>bit-fast|lean` + famille `turbo/`, tout « à mesurer ».

## Pitfalls
Numérotés comme `plan/11-pieges.md`.
- [Plantage MLX entre threads dans l'app](pitfalls/mlx-cross-thread-app-crash.md) — poids paresseux chargés sur un thread, évalués sur un autre ; tout matérialiser au chargement, un seul thread MLX (`MLXWorker`).
- [Un splat est une coquille](pitfalls/splat-is-a-shell-not-a-solid.md) — export STL : intérieur vide, crâne fait de Gaussiennes < 0,05 d'opacité, grille trop fine qui piquette ; vote de 6 rayons, toutes les Gaussiennes, 128 cellules.
- [PLY 3DGS y-bas](pitfalls/ply-y-down-convention.md) — `save_ply` écrit (x, −z, y) : le haut de l'objet sur −y ; RealityKit (y-haut) le tourne d'un demi-tour (n° 21).
- [`Float.pi` arrondi vers zéro](pitfalls/swift-float-pi-rounded-toward-zero.md) — un ulp sous le π de torch, 2,4e-4 sur sin/cos à 3000 rad ; `Float(Double.pi)` ; l'`exp` de torch à ~1 ulp impose une tolérance trig de 3e-4 (n° 20).

## Investigations
Rien pour l'instant.

## Transmissions et mémoire
- [Mémoire d'agent](agent-memory.md) — conventions de Vincent, état du projet : à lire en début de session.
