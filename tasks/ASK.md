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
