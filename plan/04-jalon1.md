## 4. Jalon 1 — Fondations, sortie et décodeurs (latent Python → `.ply` Swift)

Pourquoi en premier : c'est la moitié la plus vérifiable du pipeline. Un latent [1,8192,16]
produit par Python suffit à tester octree + Gaussiennes + PLY contre `save_ply`, sans encodeur
ni débruiteur.

| Fiche | Objet | Porte |
|---|---|---|
| T-1.0 | Squelette, scripts, CLAUDE.md, sortie (`GaussianCloud`, `PLYWriter`, `CameraPose`), `FlowSchedule`, `SystematicSampler`, contrats ⛔ G-0 | `BUILD OK`, `TESTS OK (19)` — **fait** |
| T-1.1 | Environnement Python de référence (`.venv-ref`, idol-sdk en editable, CPU/MPS) + `Scripts/reference/run_sdk.py` (une image → `.ply`, `_input.png`, `_camera.json`) | `.ply` produit sur `assets/gallery/avocado.webp` |
| T-1.2 | Téléchargement des poids (`HubDownloader` : reprise HTTP Range, SHA-256 contre `HubFile.sha256`, `HF_TOKEN` pour le gated) + `idol download` | `idol info` → `ok` sur les 4 fichiers idol |
| T-1.3 | Fixtures tiny (Python, poids aléatoires graine fixe, dimensions réduites) pour `MultiHeadRMSNorm`, `LayerNorm32`, `PcdAbsolutePositionEmbedderv2`, `SinusoidalEmbedder`, `MultiHeadAttention` (self/cross), deux blocs de décodeur + `Scripts/check-fixtures.sh` + `check-parity.sh` | `FIXTURES OK` |
| T-1.4 | Couches communes Swift (`Models/Layers/`) + parité tiny | `PARITY OK layers` (rel ≤ 1e-5 fp32) |
| T-1.5 | Décodeur octree Swift + descente (bruit rejoué) + parité tiny puis real | ancrages identiques au bit (même bruit) |
| T-1.6 | Décodeur gaussien + `to_representation` en MLX (contre `GaussianCloud.activate`) | parité real : rel ≤ 1e-3 fp16 |
| T-1.7 | `idol decode --latent latent.npy` → `.ply` ; comparaison au `.ply` Python ⛔ G-1 | `PARITY OK decode` + Vincent ouvre les deux `.ply` |

Fixtures real (latent, bruit des 8 niveaux, ancrages, sorties brutes) : générées par
`Scripts/reference/real_fixtures.py` sous `$SPERID_MODELS_DIR/parity/` (jamais committées).
