# Questions à Vincent

Gabarit dans `AGENTS.md` §6. Une réponse est datée et recopiée sous la question.

## ASK — T-1.0 ⛔ G-0 — 2026-11-15
- Contexte : le squelette est vert (`BUILD OK`, `TESTS OK (19)`). Idol lit DINOv3 ViT-H+/16, dépôt **gated** : ton compte HF n'y a pas accès (`HF_FS_ACCESS_DENIED` le 2026-11-15). FLUX.2-dev t'est accessible. `Comfy-Org/Idol` redistribue DINOv3 (bf16) et le VAE sans gate, sous l'étiquette apache-2.0, ce qui ne correspond pas à la licence DINOv3 de Meta.
- Ce que j'ai essayé : lecture des en-têtes Comfy (615 tenseurs, format transformers) ; le checkpoint local `FLUX.2-small-decoder/full_encoder_small_decoder.safetensors` porte les mêmes 111 clés d'encodeur que le VAE Comfy (valeurs à comparer en T-3.3).
- Question 1 : source de DINOv3 ?
- Options : A) tu demandes l'accès sur https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m (recommandé ; les Jalons 1-2 n'en dépendent pas, l'approbation a le temps d'arriver) ; B) repack `Comfy-Org/Idol` par défaut ; C) A par défaut, B en repli explicite (`--dinov3-source comfy`).
- Question 2 : nom du dépôt / paquet / CLI : `sperid-swift-mlx` / `SperidSwift` / `idol` — on garde ?
- Question 3 (non bloquante) : détourage au Jalon 4 — BiRefNet porté (fidèle, coûteux : Swin-L + conv déformable) ou Vision d'Apple par défaut et BiRefNet en option ?

### Réponses de Vincent — 2026-11-15
- Question 1 : **A** — accès demandé sur `facebook/dinov3-vith16plus-pretrain-lvd1689m` (en attente d'approbation). Pas de repack Comfy par défaut.
- Question 2 : **nom gardé** — `sperid-swift-mlx` / `SperidSwift` / `idol`.
- Question 3 : ouverte (à trancher avant le Jalon 4).

## ASK — T-1.7 ⛔ G-1 — 2026-11-15
- Contexte : Jalon 1 terminé. La moitié aval du pipeline (latent → octree → Gaussiennes → `.ply`) est en Swift : en fp32, ancrages identiques au bit à Python et décodeur gaussien à 5e-6 ; en fp16 (précision de référence), Swift est 5× plus proche du fp32 que PyTorch MPS ; le `.ply` écrit est à 1,9e-6 de `save_ply`.
- Ce que j'ai essayé : faute d'image encodée au moment des fixtures, le latent vient du **vrai débruiteur sur la branche non conditionnée** (condition nulle) : un objet « moyen », pas un objet d'une photo.
- Question : ouvres-tu les deux fichiers dans un viewer 3DGS (SuperSplat, https://superspl.at/editor, glisser-déposer) et confirmes-tu qu'ils montrent le même objet ?
  - Python : `~/Library/Caches/models/parity/uncond.ply`
  - Swift : `.local-runs/parity/decode_swift_fp16.ply` (dans le dépôt)
- Options : A) même objet, on passe au Jalon 2 (débruiteur) ; B) différence visible (décrire).
- Rappel, non bloquant : Q3 de G-0 (détourage BiRefNet ou Vision d'Apple) reste ouverte pour le Jalon 4.

### Réponse de Vincent — 2026-11-15
- G-1 : **A** — même objet. On passe au Jalon 2, **avec de vrais objets** désormais : les fixtures du débruiteur et des étapes suivantes partent d'images de la galerie (DINOv3 accessible), plus de latent non conditionné.
