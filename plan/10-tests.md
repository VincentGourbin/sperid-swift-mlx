## 10. Stratégie de tests et de parité

### Trois tiers
1. **Tier 1, sans poids** (toujours vert, `Scripts/check-tests.sh`) : formules pures (grille,
   systématique, activations, PLY, caméra, contrats d'en-têtes), puis fixtures **tiny** sous
   `parity/` (poids aléatoires, dimensions réduites, générées par les classes **amont** de
   `reference/idol-sdk`, jamais par une réimplémentation ; métadonnées `seed` + `upstream_sha`).
2. **Tier 2, poids réels** (`Real*Tests.swift`, gardés par `SPERID_MODELS_DIR`) : fixtures real
   sous `$SPERID_MODELS_DIR/parity/` (jamais committées), `Scripts/check-parity.sh <composant>`.
3. **Tier 3, bout en bout** : 8 images de la galerie, `.ply` Swift contre `.ply` Python, rendu de
   contrôle (T-6.1) ; revue visuelle de Vincent aux GATES.

### Le bruit est une entrée
MLX ne reproduit pas le flux de `torch.manual_seed`. Chaque fixture real enregistre **tous**
les tirages (`plan/02.1`) — ε du VAE, bruit `latent`/`camera`, u0 de chaque cellule à chaque
niveau, jitter final — et le test Swift les rejoue par un `NoiseSource` de rejeu. Sans cela,
la descente d'octree diverge dès le premier niveau et aucune comparaison n'a de sens.
Avec le même bruit, les **ancrages sont comparables au bit** (calcul entier/Float sur CPU) tant que
les probabilités par cellule restent dans la tolérance : la porte de T-1.5 compte les cellules
dont le partage diffère (attendu 0 en fp32, ≤ 0,1 % en fp16).

### Tolérances de départ (à resserrer par mesure)
| Composant | dtype | Métrique | Seuil |
|---|---|---|---|
| Couches tiny | fp32 | rel max | 1e-5 |
| Débruiteur, un forward | fp16 | rel L2 | 5e-3 |
| Latent final 50 pas | fp16 | rel L2 | 2e-2 |
| DINOv3 | bf16 | cos par jeton | ≥ 0,999 |
| Décodeur gaussien | fp16 | rel L2 sorties brutes | 1e-3 |
| Bout en bout | — | PSNR rendu | ≥ 35 dB |

### Pièges d'instrument (hérités de YuE2)
`xcodebuild test` relance un worker planté et affiche ✔ : `Scripts/run-tests.sh` + `xcrun xctest`
pour le code de sortie ; parallélisme swift-testing à 1 (deadlock ABBA compile × vjp) ;
`xcodebuild` ne transmet que les variables `TEST_RUNNER_` (le script relaie `SPERID_*`).
