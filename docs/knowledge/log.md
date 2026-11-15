# Journal

- 2026-11-15 — **Création du dépôt** : lecture de `speridlabs/idol-sdk` @ `8db7003` (pipeline, DiT, décodeurs, BiRefNet, conditionnement, turbo.py) ; en-têtes safetensors relevés par requêtes Range (`Scripts/reference/dump_headers.py`) : denoiser 560 tenseurs F32, octree 60, gaussien 328, BiRefNet 742 (F16 + 67 I64), LoRA turbo 448, DINOv3 Comfy 615 BF16, VAE Comfy 251.
- 2026-11-15 — **Accès** : `facebook/dinov3-vith16plus-pretrain-lvd1689m` refusé au compte (HF_FS_ACCESS_DENIED) ; `black-forest-labs/FLUX.2-dev/vae` lisible.
- 2026-11-15 — **Encodeur VAE local** : `~/Library/Caches/models/black-forest-labs/FLUX.2-small-decoder/full_encoder_small_decoder.safetensors` et `Comfy-Org/Idol/vae/flux2-vae.safetensors` ont le **même ensemble** de 111 clés `encoder.*`/`quant_conv.*`/`bn.*`, mêmes formes et dtypes (comparaison complète des en-têtes) ; valeurs non comparées.
- 2026-11-15 — **Nombre de jetons d'image vérifié par la structure** : DINOv3 = CLS + 4 registres + 64² = 4101, ce qui explique les 5 lignes de zéros ajoutées au flux VAE ; séquence jointe du débruiteur 12 294.
- 2026-11-15 — **Faits DINOv3** lus dans transformers `3a134bc` : ViT-H+ = 1280, 32 couches, 20 têtes, gated SiLU 5120, `rope_theta` 100, RoPE `rotate_half` sur les patches seulement, `layer_norm_eps` 1e-5, pas de biais de clé.
- 2026-11-15 — **Écarts de la référence notés** : turbo 8 pas (docstring `turbo.py`) contre 5 (carte) ; `num_gaussians` « multiple of 32 » (README) mais arrondi par le code ; `systematic_sample` divise par `clamp_min(sum, 1)`.
- 2026-11-15 — T-1.0 validée : `BUILD OK`, `TESTS OK (19 tests, 0 skippés)`.
