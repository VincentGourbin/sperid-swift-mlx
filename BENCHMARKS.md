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
