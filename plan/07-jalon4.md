## 7. Jalon 4 — Prétraitement, BiRefNet, pipeline public

| Fiche | Objet | Porte |
|---|---|---|
| T-4.1 | Recadrage (`Preprocess/`) : côté court, érosion 1 px, boîte, carré × marge, composite noir ; Lanczos via vImage | sur une image RGBA : IoU masque ≥ 0,99, écart moyen ≤ 1/255 contre `_input.png` |
| T-4.2 | Protocole `BackgroundRemover` + implémentation Vision (macOS 14+/iOS 17+) comme repli | détourage des 8 images de la galerie, revue visuelle |
| T-4.3 | Swin-L (fenêtres décalées, biais de position relative) | parité real des 4 cartes de caractéristiques |
| T-4.4 | Convolution déformable (bilinéaire + matmul, puis noyau Metal si > 20 % du temps) | parité tiny contre `torchvision.ops.deform_conv2d` |
| T-4.5 | Décodeur BiRefNet + `remove_background` | IoU ≥ 0,98 contre l'alpha Python sur la galerie |
| T-4.6 | `IdolPipeline` (API publique : `generate(image:request:progress:)`, annulation, résidence par étape), `idol generate` ⛔ G-4 | 8 images de bout en bout ; critère de `plan/01-perimetre.md` |
