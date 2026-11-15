# Mémoire d'agent

## Conventions de Vincent (reprises de YuE2)
- Documents en français (plan, fiches, base de connaissances) ; code, commentaires et documentation
  utilisateur (`README.md`, `docs/*.md`) en anglais.
- Un fichier `plan/` par section, `PLAN.md` = sommaire ; fiches `tasks/T-x.y.md` à porte chiffrée ;
  ⛔ GATE = Vincent teste lui-même avant la suite.
- Copier depuis les dépôts frères (`reference/`), jamais de dépendance lourde ; mlx-swift épinglé `exact`.
- Aucun chiffre sans profiler et protocole ; un levier < 5 % est retiré avec sa mesure.
- Rien de publié (upload HF, release) sans accord explicite. Dépôt GitHub **public** depuis le 2026-11-15 (0.3.0).

## État (2026-11-15)
- Jalon 1 validé (T-1.0 à T-1.7) ; GATE G-1 posée (revue visuelle de deux `.ply`).
- DINOv3 : accès accordé, poids dans le cache HF. Détourage (Q3 de G-0) encore ouvert.
- Les fixtures real des décodeurs utilisent un latent de la branche non conditionnée du vrai débruiteur.

## Pièges durables
- Le SDK refuse de démarrer sans accès aux encodeurs gated (`require_encoder_access`) : les fixtures
  real des décodeurs chargent les décodeurs seuls.
- RoPE du débruiteur : avant les normes, paires entrelacées ; DINOv3 : `rotate_half`. Voir `plan/11-pieges.md`.
