## 3. Budget mémoire et performance attendue

Tout est **CALCUL** (dérivé des dimensions), rien n'est mesuré. Chaque chiffre a sa fiche de
mesure ; un chiffre mesuré remplace la ligne et cite `docs/knowledge/log.md`.

### 3.1 Poids (fp16/bf16 résidents)
| Bloc | Paramètres | fp16 | 8 bits (CALCUL) |
|---|---:|---:|---:|
| Débruiteur | 370,6 M | 0,74 Go | 0,40 Go |
| Décodeur octree | 50,8 M | 0,10 Go | 0,06 Go |
| Décodeur gaussien | 237,3 M | 0,47 Go | 0,26 Go |
| DINOv3 ViT-H+ | 840,6 M | 1,68 Go | 0,90 Go |
| Encodeur VAE FLUX.2 | ≈ 34 M (encodeur seul) | 0,07 Go | — (non quantifié) |
| BiRefNet | 220,7 M | 0,44 Go | — |
| **Total** | ≈ 1,75 G | **≈ 3,5 Go** | ≈ 2,1 Go |

Avec résidence par étape (BiRefNet libéré après le recadrage, encodeurs libérés avant le
débruitage), le pic de poids tombe au max des étapes : ≈ 2,2 Go (encodeurs) puis ≈ 1,3 Go
(débruiteur + décodeurs).

### 3.2 Calcul par étape (M3 Max)
Hypothèse de débit **non mesurée** : 7 à 15 TFLOPS fp16 effectifs (7-9 TFLOPS = préfill borné calcul
mesuré sur gemma-4-swift-mlx en 4 bits, catalogue `mlx-swift-audit` IT-8 ; 15 = ordre de grandeur
d'un GEMM fp16 dense). La première mesure (T-2.5) fixe la valeur et remplace la fourchette.
| Étape | FLOPs (CALCUL) | Temps attendu |
|---|---:|---:|
| Un forward du débruiteur, L = 12 294 : Linear 7,4 T + attention 14,9 T | ≈ 22 TFLOP | 1,5-3 s |
| 50 pas × CFG (100 forwards) | ≈ 2,2 PFLOP | **150-300 s** |
| Turbo : 5 forwards | ≈ 110 TFLOP | **8-16 s** |
| DINOv3, L = 4101, 32 couches | ≈ 10 TFLOP | 0,7-1,4 s |
| Décodeur gaussien, 8192 ancrages (262 k) | ≈ 12 TFLOP | 0,8-1,7 s |
| Décodeur gaussien, 32 768 ancrages (1 M) | ≈ 110 TFLOP | 7-16 s |

Lecture : le débruiteur domine à 95 % en mode de base ; **le turbo est le levier n° 1** (×20 en
évaluations) et c'est lui qui rend l'iPhone plausible (A17 Pro ≈ 2 TFLOPS effectifs ⇒ ≈ 1 min en
turbo, CALCUL). Pour comparaison, la carte donne 6,8 s sur H100 et 19,1 s sur RTX 4090 (50 pas).

### 3.3 Activations : l'attention décide fast / lean
- Débruiteur : scores 16 têtes × 12 294² en fp16 = **4,8 Go par forward** s'ils sont matérialisés
  (×2 en CFG batché). Le SDPA fusionné de MLX (head_dim 64) ne devrait pas les matérialiser :
  **à mesurer en premier** (T-2.5), c'est ce qui choisit CFG batché (`fast`) contre CFG séquentiel
  et requêtes tranchées (`lean`).
- Décodeur gaussien à 1 M : 32 768² × 16 têtes = **34 Go** s'ils étaient matérialisés.
- Tenseurs de MLP du débruiteur : 12 294 × 4096 × 2 o ≈ 100 Mo, négligeables.

### 3.4 Cibles (portes des Jalons) — temps à fixer après T-2.5
- Mac `16bit-fast` : temps de la base et du turbo **fixés après la mesure de T-2.5** (débit réel) ;
  pic ≤ 12 Go à 262 k Gaussiennes.
- iPhone 15 Pro Max `8bit-lean` turbo : pic process ≤ 4 Go (sans entitlement mémoire) ; temps fixé
  après la première mesure sur l'appareil (T-6.6).
