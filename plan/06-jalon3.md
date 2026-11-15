## 6. Jalon 3 — Encodeurs d'image (pixels → cond Swift)

| Fiche | Objet | Porte |
|---|---|---|
| T-3.1 | Décision de source DINOv3 (G-0) appliquée ; relire les clés du dépôt choisi | contrat vert sur le fichier local |
| T-3.2 | DINOv3 ViT-H+/16 (patch embed, registres, RoPE 2D `rotate_half`, layerscale, gated SiLU) + layer_norm final d'Idol | parité real `feature1` : cos ≥ 0,999 par jeton (bf16) |
| T-3.3 | Encodeur VAE FLUX.2 copié de `flux-2-swift-mlx` (encodeur seul) ; vérification des valeurs (Comfy = local `FLUX.2-small-decoder` = `FLUX.2-dev`) | SHA par tenseur, `PARITY OK vae-encoder` (mean exacte, posterior rejoué) |
| T-3.4 | Patchification 2×2, normalisation BN (eps 1e-4), padding 5 → `feature2` | parité real `feature2` rel ≤ 1e-2 (bf16) |
| T-3.5 | `ImageConditioner` (les deux flux, résidence : libérés après usage en `lean`) + `idol encode` ⛔ G-3 | pipeline `_input.png` → `.ply` entièrement Swift, comparé au Python |
