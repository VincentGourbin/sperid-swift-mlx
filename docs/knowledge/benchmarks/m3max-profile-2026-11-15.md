# Profil M3 Max 96 Go — pipeline 100 % Swift (2026-11-15)

**Conditions** : build Release, `idol generate --profile` (swift-mlx-profiler 1.5.x, `fineGrained`), image
`fig-fox.webp` du Space, 262 144 Gaussiennes, 120 s de refroidissement. Sauvegarde Time Machine (`backupd`
15-50 % CPU) active pendant les campagnes A/B/B/A : chiffres **en session**, ordres de grandeur fiables.

## Phases (turbo, 5 pas)
| Phase | Temps | GPU | CPU |
|---|---:|---:|---:|
| Chargement | 1,1 s | 85 % | 88 % |
| DINOv3 | 1,2 s | 100 % | 7 % |
| Encodeur VAE | 0,84 s | 99 % | 76 % |
| Débruitage (5 pas) | 19,2 s → 16,9 s avec la fusion | 100 % | 8 % |
| Octree (8 niveaux) | 0,6 s | 100 % | 20 % |
| Décodeur gaussien | 1,75 s | 100 % | 5 % |
| Prétraitement + assemblage + écriture | 0,23 s | 35-45 % | 90 % |
| **Total** | **24,9 s → 22,2 s** | | |

Base (50 pas, CFG) : 7 min 50 s, 9,28 s/pas (σ 1,05 s, min 6,9, max 11,7 : bridage thermique sur 8 min).

## Un bloc joint du débruiteur (12 294 jetons, fp16, `idol profile denoiser-ops`)
SDPA 79 ms (59 %), MLP 25,6 ms (19 %), rotation + RMSNorm 13,7 ms (10 %), qkv 8,5 ms (6 %), RePo 3,6 ms,
reste 4,4 ms ; bloc 134 ms. Attention et GEMM ≈ 8-9 TFLOPS : borné calcul.

## Leviers
| Levier | Mesure | Décision |
|---|---|---|
| Rotation + RMSNorm compilées (`fusedRotaryNorm`) | A/B/B/A bloc : A1 137,3 (hors bande, SDPA 122 ms), B 118,6 / 118,6, A2 123,6 ms → −4,0 % ; bit-identique (parité inchangée) | **gardé** (levier exact : décision de Vincent, seuil de 5 % levé pour l'exact) |
| Cache MLX plafonné 2 Go + vidé entre étapes | turbo : 22,2 s / pic process 33,8 Go → 22,7 s / **11,3 Go** | **gardé** par défaut dans le CLI et l'app (la bibliothèque reste neutre) |
| Éval de l'encodeur VAE bloc par bloc | pic MLX actif inchangé (11,0 Go) | gardé (sans coût), cause ailleurs |
| Fusions de modulation / résiduel-porte | CALCUL < 1 % du bloc (passages mémoire de 25 Mo) | non faites |

## Mémoire : le pic est dans l'encodeur VAE (`idol profile vae-peaks`)
Un bloc résiduel à 1024² : **8,3 Go** (convolution 3×3 dépliée, ≈ 4,8 Go d'im2col fp32 + copies GroupNorm) ;
512² : 4,0-4,3 Go ; 256² : 2,2-2,5 Go ; attention centrale (16 384 × 512) : 1,5 Go. Correctif prévu pour
l'iPhone : convolutions pleine résolution en bandes (halo d'un pixel), statistiques GroupNorm globales.
