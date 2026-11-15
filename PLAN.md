# sperid-swift-mlx — Plan d'implémentation

> **Statut : rév. 1 du 2026-11-15 — plan initial**, rédigé après lecture intégrale de la référence Python
> (`speridlabs/idol-sdk` @ `8db7003`, 1 600 lignes utiles), des en-têtes safetensors publiés et de la
> source DINOv3 de transformers, et après inventaire des briques des dépôts frères (YuE2, FLUX.2).
> Roadmap : **Jalon 1 (décodeurs) → 2 (débruiteur) → 3 (encodeurs) → 4 (prétraitement, pipeline) →
> 5 (turbo) → 6 (profils, packs, iPhone)**. À chaque GATE, Vincent teste lui-même.
> Cible : MacBook Pro M3 Max 96 Go, macOS 27, Xcode 27 (Swift 6.4) ; puis iPhone 15 Pro Max.
> Poids sous `$SPERID_MODELS_DIR` (défaut `~/Library/Caches/models`), jamais de chemin absolu dans le dépôt.

---

## Sommaire — le plan vit dans `plan/`, un fichier par section

**Ce fichier est un sommaire. Ouvrir le fichier de la section demandée, un seul à la fois.**
Les fiches (`tasks/T-x.y.md`) citent ces fichiers par nom.

| Fichier | Section | Mots |
|---|---|---:|
| `plan/00-mode-execution.md` | 0. Mode d'exécution | 255 |
| `plan/01-perimetre.md` | 1. Objectif et périmètre | 256 |
| `plan/02.1-pipeline.md` | 2.1 Le pipeline, étape par étape (`idol/pipeline.py`, idol-sdk `8db7003`) | 364 |
| `plan/02.2-denoiseur.md` | 2.2 Débruiteur `LatentSeqMMFlowModel` (`models/dit.py`, `models/layers.py`) | 497 |
| `plan/02.3-decodeurs.md` | 2.3 Décodeurs (`models/decoder.py`) | 392 |
| `plan/02.4-encodeurs.md` | 2.4 Encodeurs d'image (`conditioning.py`) | 287 |
| `plan/02.5-pretraitement.md` | 2.5 Prétraitement et détourage (`preprocess.py`, `models/birefnet.py`) | 257 |
| `plan/02.6-tenseurs.md` | 2.6 Noms de tenseurs (contrat du loader) | 293 |
| `plan/02.7-dtypes.md` | 2.7 Ordre des dtypes (à respecter pour la parité) | 197 |
| `plan/02.8-turbo.md` | 2.8 Idol Turbo (`speridlabs/idol-turbo`, `turbo.py`) | 196 |
| `plan/03-budget.md` | 3. Budget mémoire et performance attendue | 570 |
| `plan/04-jalon1.md` | 4. Jalon 1 — Fondations, sortie et décodeurs (latent Python → `.ply` Swift) | 274 |
| `plan/05-jalon2.md` | 5. Jalon 2 — Débruiteur et échantillonneur (cond Python → latent Swift) | 188 |
| `plan/06-jalon3.md` | 6. Jalon 3 — Encodeurs d'image (pixels → cond Swift) | 158 |
| `plan/07-jalon4.md` | 7. Jalon 4 — Prétraitement, BiRefNet, pipeline public | 167 |
| `plan/08-jalon5.md` | 8. Jalon 5 — Turbo | 87 |
| `plan/09-jalon6.md` | 9. Jalon 6 — Profils de référence, quantification, packs, iPhone | 168 |
| `plan/10-tests.md` | 10. Stratégie de tests et de parité | 350 |
| `plan/11-pieges.md` | 11. Checklist de pièges (contrat de revue — cocher à chaque fiche, 19 points) | 426 |
| `plan/12-ecosysteme.md` | 12. Écosystème — brique → dépôt → fichier | 222 |
| `plan/13-risques.md` | 13. Risques | 249 |
| `plan/14-decisions.md` | 14. Décisions actées et questions ouvertes | 232 |
| `plan/15-famille-sperid.md` | 15. Les autres modèles Speri Labs (préparation, pas de code avant G-6) | 162 |

Pour lire le plan d'un bloc (humain seulement) : `Scripts/plan-concat.sh | less`.
