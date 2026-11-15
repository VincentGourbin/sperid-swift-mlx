## 1. Objectif et périmètre

**Objectif** : porter `speridlabs/idol` (image → splat de Gaussiennes 3D, Apache-2.0) en Swift/MLX,
sans Python à l'inférence, sur Mac (M3 Max 96 Go) puis iPhone (A17 Pro 8 Go), avec la même
exigence que YuE2 : parité mesurée avec la référence, profils `<bits>bit-fast|lean` mesurés,
documentation utilisateur en anglais et base de connaissances OKF en français.

**Dans le périmètre**
1. Pipeline complet : prétraitement (BiRefNet), encodeurs (DINOv3 ViT-H+/16, encodeur VAE FLUX.2),
   débruiteur `LatentSeqMMFlowModel` (Euler + CFG), décodeur octree, décodeur gaussien,
   écriture `.ply` 3DGS et caméra JSON — sorties identiques à `python -m idol.sample`.
2. `idol-turbo` : LoRA de phase (slow / fast), 5 pas sans guidage.
3. Bibliothèque `IdolCore`, CLI `idol`, API documentée pour une app (FluxForge Studio, app iOS).
4. Profils de référence et packs pré-quantifiés (SHA-256), publiés sur HF après accord.

**Hors périmètre (pour l'instant)**
- Rendu des splats (le `.ply` s'ouvre dans tout viewer 3DGS ; un viewer Metal est une option
  pour l'app, `plan/14-decisions.md` Q-4).
- Entraînement ou distillation (le turbo est consommé, pas entraîné).
- `iris-3b` et `iris-z` : architecture prévue (`plan/15-famille-sperid.md`), aucun code avant G-6.
- Démo Gradio, `torch.compile`, backends CUDA : sans équivalent à porter.

**Critère de réussite du Jalon final** : sur les 8 images de `reference/idol-sdk/assets/gallery`,
le `.ply` Swift `16bit-fast` et le `.ply` Python à graine et bruit identiques (bruit rejoué)
donnent une PSNR ≥ 35 dB en rendu de la vue d'entrée et des 7 vues du banc (rendu de contrôle
par `gsplat` côté Python), et Vincent valide visuellement dans un viewer.
