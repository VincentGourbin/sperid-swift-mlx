# Benchmarks

Mesures reproductibles du port Idol → MLX Swift sur Apple Silicon. **Aucune mesure pour l'instant** :
les coûts attendus (CALCUL) sont dans `plan/03-budget.md`.

## Comment lire ce fichier

Source unique des chiffres : `swift-mlx-profiler`, jamais un chronométrage à la main. Une ligne =
un run de `idol generate --profile` (ou d'une étape seule : `idol denoise|decode --profile`).

## Comment contribuer une ligne

1. Reproduire avec la commande citée dans l'entrée `tasks/JOURNAL.md` de la fiche (build Release, GPU
   refroidi, A/B/B/A — `CLAUDE.md` § Performance work).
2. Ajouter une ligne : date, SHA court, profil, image, pas, Gaussiennes, temps par étape, pic MLX,
   `phys_footprint`, chemin de la trace (sous `.local-runs/`, jamais committée).
3. Ne jamais modifier une ligne existante.

| Date | Commit | Machine | Profil | Image | Pas | Gaussiennes | Encodeurs s | Débruitage s | Décodage s | Total s | Pic Go | Trace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-15 19:36 | 5088eaa | M3 Max 96 Go | denoiser fp16, lot 1 (A1) | aléatoire, formes réelles | 1 forward × 3 | — | — | 5,76 ± 0,31 | — | — | 2,06 MLX / 4,98 process | `.local-runs/bench.noindex/profile-denoiser/trace-b1.json` |
| 2026-11-15 19:38 | 5088eaa | M3 Max 96 Go | denoiser fp16, lot 2 (B1) | idem | 1 forward × 3 | — | — | 7,21 ± 0,25 (3,61 / échantillon) | — | — | 2,75 MLX / 7,28 process | idem, trace-b2.json |
| 2026-11-15 19:40 | 5088eaa | M3 Max 96 Go | denoiser fp16, lot 2 (B2) | idem | 1 forward × 3 | — | — | 7,03 ± 0,55 (3,52 / échantillon) | — | — | 2,71 MLX / 7,28 process | idem |
| 2026-11-15 19:42 | 5088eaa | M3 Max 96 Go | denoiser fp16, lot 1 (A2) | idem | 1 forward × 3 | — | — | 3,36 ± 0,11 | — | — | 2,06 MLX / 4,98 process | idem |
