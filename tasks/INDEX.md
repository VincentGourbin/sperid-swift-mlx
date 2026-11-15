# Fiches — ordre imposé (une fiche = un commit = une validation)

État : `à faire` · `en cours` · `validée` · `bloquée`. Une fiche ne démarre que si tous ses prérequis sont `validée`. ⛔ = se termine par une question dans `tasks/ASK.md`. Le détail (porte) est dans le fichier de jalon cité ; une fiche n'a son fichier `T-x.y.md` qu'au moment où son jalon s'ouvre.

| Fiche | Titre | Jalon (plan) | Prérequis | État |
|---|---|---|---|---|
| T-1.0 | Squelette, scripts, CLAUDE.md, sortie (`GaussianCloud`, `PLYWriter`, `CameraPose`), `FlowSchedule`, `SystematicSampler`, contrats ⛔ G-0 | `plan/04-jalon1.md` | — | validée (G-0 levée 2026-11-15) |
| T-1.1 | Environnement Python de référence (`.venv-ref`, idol-sdk en editable, CPU/MPS) + `Scripts/reference/run_sdk.py` (une image → `.ply`, `_input.png`, `_camera.json`) | `plan/04-jalon1.md` | T-1.0 | validée (passe SDK complète après accès DINOv3) |
| T-1.2 | Téléchargement des poids (`HubDownloader` : reprise HTTP Range, SHA-256 contre `HubFile.sha256`, `HF_TOKEN` pour le gated) + `idol download` | `plan/04-jalon1.md` | T-1.0 | validée |
| T-1.3 | Fixtures tiny (Python, poids aléatoires graine fixe, dimensions réduites) pour `MultiHeadRMSNorm`, `LayerNorm32`, `PcdAbsolutePositionEmbedderv2`, `SinusoidalEmbedder`, `MultiHeadAttention` (self/cross), deux blocs de décodeur + `Scripts/check-fixtures.sh` + `check-parity.sh` | `plan/04-jalon1.md` | T-1.1 | validée |
| T-1.4 | Couches communes Swift (`Models/Layers/`) + parité tiny | `plan/04-jalon1.md` | T-1.3 | validée |
| T-1.5 | Décodeur octree Swift + descente (bruit rejoué) + parité tiny puis real | `plan/04-jalon1.md` | T-1.4, T-1.2 | validée |
| T-1.6 | Décodeur gaussien + `to_representation` en MLX (contre `GaussianCloud.activate`) | `plan/04-jalon1.md` | T-1.5 | validée |
| T-1.7 | `idol decode --latent latent.npy` → `.ply` ; comparaison au `.ply` Python ⛔ G-1 | `plan/04-jalon1.md` | T-1.6 | validée (GATE G-1 posée) |
| T-2.1 | `SinusoidalEmbedder` (cos/sin, t·1000), `PcdAbsolutePositionEmbedder` v1, MLP caméra | `plan/05-jalon2.md` | T-1.7 | à faire |
| T-2.2 | `RePo3DRotaryEmbedding` + rotation par paires entrelacées (piège 1-2) | `plan/05-jalon2.md` | T-2.1 | à faire |
| T-2.3 | `RopeMultiHeadAttention`, `UnifiedTransformerBlock` (modulé / non modulé) | `plan/05-jalon2.md` | T-2.2 | à faire |
| T-2.4 | `LatentSeqMMFlowModel` complet + loader (560 tenseurs, fp16 des blocs, fp32 ailleurs) | `plan/05-jalon2.md` | T-2.3 | à faire |
| T-2.5 | **Mesure** : un forward L = 12 294, lot 1 et 2 — temps, pic MLX, SDPA fusionné ou non | `plan/05-jalon2.md` | T-2.4 | à faire |
| T-2.6 | Échantillonneur Euler + CFG (séquentiel et batché), annulable, une synchro par pas (IT-2) | `plan/05-jalon2.md` | T-2.5 | à faire |
| T-2.7 | `idol denoise --cond cond.safetensors` → latent ; chaîne avec T-1.7 → `.ply` ⛔ G-2 | `plan/05-jalon2.md` | T-2.6 | à faire |
| T-3.1 | Décision de source DINOv3 (G-0) appliquée ; relire les clés du dépôt choisi | `plan/06-jalon3.md` | T-2.7, G-0 | à faire |
| T-3.2 | DINOv3 ViT-H+/16 (patch embed, registres, RoPE 2D `rotate_half`, layerscale, gated SiLU) + layer_norm final d'Idol | `plan/06-jalon3.md` | T-3.1 | à faire |
| T-3.3 | Encodeur VAE FLUX.2 copié de `flux-2-swift-mlx` (encodeur seul) ; vérification des valeurs (Comfy = local `FLUX.2-small-decoder` = `FLUX.2-dev`) | `plan/06-jalon3.md` | T-3.2 | à faire |
| T-3.4 | Patchification 2×2, normalisation BN (eps 1e-4), padding 5 → `feature2` | `plan/06-jalon3.md` | T-3.3 | à faire |
| T-3.5 | `ImageConditioner` (les deux flux, résidence : libérés après usage en `lean`) + `idol encode` ⛔ G-3 | `plan/06-jalon3.md` | T-3.4 | à faire |
| T-4.1 | Recadrage (`Preprocess/`) : côté court, érosion 1 px, boîte, carré × marge, composite noir ; Lanczos via vImage | `plan/07-jalon4.md` | T-3.5 | à faire |
| T-4.2 | Protocole `BackgroundRemover` + implémentation Vision (macOS 14+/iOS 17+) comme repli | `plan/07-jalon4.md` | T-4.1 | à faire |
| T-4.3 | Swin-L (fenêtres décalées, biais de position relative) | `plan/07-jalon4.md` | T-4.2 | à faire |
| T-4.4 | Convolution déformable (bilinéaire + matmul, puis noyau Metal si > 20 % du temps) | `plan/07-jalon4.md` | T-4.3 | à faire |
| T-4.5 | Décodeur BiRefNet + `remove_background` | `plan/07-jalon4.md` | T-4.4 | à faire |
| T-4.6 | `IdolPipeline` (API publique : `generate(image:request:progress:)`, annulation, résidence par étape), `idol generate` ⛔ G-4 | `plan/07-jalon4.md` | T-4.5 | à faire |
| T-5.1 | Chargeur LoRA de phase (448 tenseurs) et `LoRALinear` à deux experts | `plan/08-jalon5.md` | T-4.6 | à faire |
| T-5.2 | Fusion par phase (deux jeux fusionnés) contre delta à la volée : mesure | `plan/08-jalon5.md` | T-5.1 | à faire |
| T-5.3 | `IdolRequest.turbo`, `idol generate --turbo` ⛔ G-5 | `plan/08-jalon5.md` | T-5.2 | à faire |
| T-6.1 | Outil de qualité : rendu de contrôle (PSNR/SSIM à la vue d'entrée + 7 vues) côté Python sur deux `.ply` | `plan/09-jalon6.md` | T-5.3 | à faire |
| T-6.2 | Baseline mesurée `16bit-fast` (base et turbo), protocole `plan/00` | `plan/09-jalon6.md` | T-6.1 | à faire |
| T-6.3 | Quantification par voie (débruiteur 8 bits, décodeurs 8 bits, DINOv3 8 bits ; 4 bits mesuré, attendu rejeté pour l'étape itérée, IT-7) ; couverture en octets (IT-5) | `plan/09-jalon6.md` | T-6.2 | à faire |
| T-6.4 | `IdolReferenceProfile` (`<bits>bit-fast\|lean`, `turbo/<bits>bit-*`), `idol references`, `--reference` | `plan/09-jalon6.md` | T-6.3 | à faire |
| T-6.5 | Packs pré-quantifiés (`int8-mixed`…) + SHA-256 + `Scripts/publish-packs.sh` (n'uploade rien) ⛔ G-6 | `plan/09-jalon6.md` | T-6.4 | à faire |
| T-6.6 | Build iOS, mesure iPhone 15 Pro Max (`8bit-lean` turbo), porte GPU / annulation / checkpoint au pas (IT-12) | `plan/09-jalon6.md` | T-6.5 | à faire |
