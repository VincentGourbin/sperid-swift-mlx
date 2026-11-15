# Décision — source des poids des encodeurs (DINOv3, VAE FLUX.2)

**Contexte** (2026-11-15) : Idol lit deux encodeurs gated, `facebook/dinov3-vith16plus-pretrain-lvd1689m`
et `black-forest-labs/FLUX.2-dev` (sous-dossier `vae`). Le compte HF de Vincent lit FLUX.2-dev mais
**pas** DINOv3.

**Constat**
- `Comfy-Org/Idol` @ `e907d8d` redistribue `clip_vision/dino_v3_vit_h_bf16.safetensors` (615 tenseurs,
  clés au format transformers) et `vae/flux2-vae.safetensors`, sans gate, sous l'étiquette apache-2.0 —
  étiquette qui ne reflète pas la licence DINOv3 de Meta.
- L'encodeur FLUX.2 existe aussi en local (`FLUX.2-small-decoder/full_encoder_small_decoder.safetensors`,
  mêmes 111 clés que le VAE Comfy).

**Décision (G-0, 2026-11-15)** : option A retenue par Vincent — dépôt officiel, accès demandé, en attente d'approbation. Proposition initiale : option C — dépôt officiel par défaut (Vincent demande
l'accès ; les Jalons 1-2 n'en dépendent pas), repack Comfy en repli **explicite** (option CLI), jamais
silencieux. Pour le VAE : `FLUX.2-dev` (accessible) par défaut ; le fichier local sert à vérifier
l'égalité des valeurs (T-3.3).

**Non retenu** : choisir Comfy par défaut sans accord (licence) ; dépendre de `flux-2-swift-mlx` pour
charger le VAE (voir `package-layout.md`).

**Mise à jour 2026-11-15** : accès accordé ; DINOv3 et `FLUX.2-dev/vae` téléchargés dans le cache HF.

**Ouvert** : format des clés du dépôt DINOv3 officiel (à relire en T-3.1).
