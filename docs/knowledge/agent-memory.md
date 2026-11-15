# Mémoire d'agent

## Conventions de Vincent (reprises de YuE2)
- Documents en français (plan, fiches, base de connaissances) ; code, commentaires et documentation
  utilisateur (`README.md`, `docs/*.md`) en anglais.
- Un fichier `plan/` par section, `PLAN.md` = sommaire ; fiches `tasks/T-x.y.md` à porte chiffrée ;
  ⛔ GATE = Vincent teste lui-même avant la suite.
- Copier depuis les dépôts frères (`reference/`), jamais de dépendance lourde ; mlx-swift épinglé `exact`.
- Aucun chiffre sans profiler et protocole ; un levier < 5 % est retiré avec sa mesure.
- Rien de publié (push public, upload HF) sans accord explicite. Dépôt GitHub **privé** pour l'instant.

## État (2026-11-15)
- T-1.0 validée ; G-0 posée (source DINOv3, nom du dépôt, détourage).
- Jalons 1-2 n'ont pas besoin de DINOv3 : on peut avancer pendant que l'accès est demandé.

## Pièges durables
- Le SDK refuse de démarrer sans accès aux encodeurs gated (`require_encoder_access`) : les fixtures
  real des décodeurs chargent les décodeurs seuls.
- RoPE du débruiteur : avant les normes, paires entrelacées ; DINOv3 : `rotate_half`. Voir `plan/11-pieges.md`.
