## 5. Jalon 2 — Débruiteur et échantillonneur (cond Python → latent Swift)

| Fiche | Objet | Porte |
|---|---|---|
| T-2.1 | `SinusoidalEmbedder` (cos/sin, t·1000), `PcdAbsolutePositionEmbedder` v1, MLP caméra | parité tiny |
| T-2.2 | `RePo3DRotaryEmbedding` + rotation par paires entrelacées (piège 1-2) | parité tiny, rel ≤ 1e-5 |
| T-2.3 | `RopeMultiHeadAttention`, `UnifiedTransformerBlock` (modulé / non modulé) | parité tiny |
| T-2.4 | `LatentSeqMMFlowModel` complet + loader (560 tenseurs, fp16 des blocs, fp32 ailleurs) | parité real d'un forward (cond + bruit + t Python) : rel ≤ 5e-3 |
| T-2.5 | **Mesure** : un forward L = 12 294, lot 1 et 2 — temps, pic MLX, SDPA fusionné ou non | ligne `BENCHMARKS.md` + décision fast/lean (`decisions/`) |
| T-2.6 | Échantillonneur Euler + CFG (séquentiel et batché), annulable, une synchro par pas (IT-2) | latent final à 50 pas : rel ≤ 2e-2 contre Python, même bruit |
| T-2.7 | `idol denoise --cond cond.safetensors` → latent ; chaîne avec T-1.7 → `.ply` ⛔ G-2 | Vincent compare les `.ply` (Python cond → Swift tout le reste) |
