## 12. Écosystème — brique → dépôt → fichier

`Scripts/link-references.sh` crée `reference/` (gitignoré). « Copier d'abord, adapter ensuite. »

| Brique | Source | Fichier | Usage |
|---|---|---|---|
| Référence Python Idol | `speridlabs/idol-sdk` @ `8db7003` (2026-11-09) | `idol/**/*.py` | vérité du port |
| DINOv3 | transformers @ `3a134bc` (sparse) | `models/dinov3_vit/modeling_dinov3_vit.py` | faits DINOv3 |
| Encodeur VAE FLUX.2 | `flux-2-swift-mlx` | `Sources/Flux2Core/VAE/{VAEEncoder,ResnetBlock,AutoencoderKL}.swift` | T-3.3 (encodeur seul) |
| Chargement safetensors, F32 → fp16 | `flux-2-swift-mlx` | `Sources/Flux2Core/Loading/` | T-1.4 |
| Téléchargeur HF reprise + SHA-256 | `YuE2-mlx-swift` | `Sources/YuE2Core/Loading/ModelDownloader.swift` | T-1.2 |
| Scripts de vérification, runner de tests | `YuE2-mlx-swift` | `Scripts/{check-*,run-tests}.sh` | faits (T-1.0) |
| Profils de référence | `YuE2-mlx-swift` | `Sources/YuE2Core/Configuration/ReferenceProfiles.swift`, `docs/References.md` | T-6.4 |
| Résidence par étape, limites mémoire | `YuE2-mlx-swift` | `docs/knowledge/decisions/stage-scoped-weight-residency.md`, `Memory/` | T-4.6, T-6.6 |
| Quantification couche par couche | `gemma-4-swift-mlx` | IT-6 (`G 50bef8fc`) | T-6.3 |
| Profilage | `swift-mlx-profiler` ≥ 1.5.1 | `MLXProfiler` | toutes les mesures |
| Catalogue des techniques | skill `mlx-swift-audit` | `references/{techniques,iterative-models,profiles-standard}.md` | Jalon 6 |
| Patterns de défauts | skill `mlx-swift-patterns` | catalogue MLX-0xx | avant chaque GATE |

Références externes à surveiller (non utilisées) : `litert-community/Idol` (port LiteRT, preuve
qu'Idol tient sur mobile), `TimeLordRaps/Idol-GGUF`, ComfyUI (implémentation TripoSplat/Idol,
autre lecture de la référence en cas de doute).
