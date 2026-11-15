# Décision — organisation du paquet

**Contexte** (2026-11-15) : Vincent veut « un framework Swift MLX pour les modèles Speri Labs, en
particulier celui qui génère en 3D », cohérent avec YuE2 en performances et en documentation.

**Décision**
- Un paquet `SperidSwift`, cibles `SperidCore` (fichiers Hub épinglés, en-têtes safetensors, contrats ;
  rien de spécifique à un modèle), `IdolCore`, CLI `idol`, tests `SperidTests`. Iris viendra en
  `IrisCore` + `iris`. Même schéma que `YuE2Core` + `SheetSage2Core`.
- Rien ne monte dans `SperidCore` avant d'avoir deux utilisateurs (échantillonneur flow-matching,
  profils, téléchargeur) : pas d'abstraction spéculative.
- Briques des dépôts frères **copiées** via `reference/` (VAE de `flux-2-swift-mlx`, téléchargeur et
  scripts de YuE2), pas de dépendance SPM : `flux-2-swift-mlx` entraîne encodeurs de texte, chaînes
  et entraînement LoRA pour trois fichiers utiles.
- Octree sur CPU (formes dépendantes des données), 8 forwards GPU ; tout tirage aléatoire via
  `NoiseSource` (parité par rejeu du bruit Python).
- mlx-swift 0.32.3 `exact` (dernière montée d'un dépôt frère, gemma-4-swift-mlx `130d1ea`) ;
  `swift-mlx-profiler` ≥ 1.5.1 (compile pour iOS).

**Mesure** : `BUILD OK`, `TESTS OK (19 tests)` au 2026-11-15.

**Ouvert** : le nom (`sperid-swift-mlx`) est soumis à G-0.
