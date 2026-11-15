## 11. Checklist de pièges (contrat de revue — cocher à chaque fiche)

1. **RoPE du débruiteur avant `q_norm`/`k_norm`** — l'inverse de YuE2. Rotation dépendante de la
   tête, angles appris (RePo), recalculés à chaque bloc depuis son entrée.
2. **Paires entrelacées** `(x[2i], x[2i+1])` pour la RoPE du débruiteur (`traditional: true`) ;
   **moitiés** (`rotate_half`, `traditional: false`) pour DINOv3. Ne pas uniformiser.
3. `clamp_mul(x, f)` = `x·f` en inférence ; ne pas porter le `tanh` seul (sortie bornée fausse).
4. **`MultiHeadRMSNorm`** = `normalize(x, dim=-1)·gamma·√64` en fp32 (norme L2, pas RMS + eps) ;
   `gamma` a la forme [têtes, 64].
5. **GELU tanh** partout dans les MLP Idol ; **SiLU gated** dans DINOv3 ; GELU exacte dans Swin.
6. Embedding de temps : **cos puis sin**, entrée **1000·t** ; octree : `max_period` 1024, 2π, sur
   la **résolution** `1 << l`.
7. Plongements de position : v1 (débruiteur) angle **2π**, fréquences `2^[0..15] ++ …` ; v2
   (décodeurs) angle **π**, `2^linspace(0,10)` ; zéro-padding à 1024 dans les deux.
8. **Eval tenseur par tenseur** pendant la conversion (F32 → fp16, transposition des conv) ;
   jamais `eval(model.parameters())` global.
9. **Condition négative = zéros avant les projections** (les biais passent) ; jamais un cond projeté nul.
10. Le VAE **échantillonne** le posterior (pas la moyenne) ; patchification `c·4 + dy·2 + dx` ;
    BN eps 1e-4 ; 5 lignes de zéros **en tête**.
11. DINOv3 : RoPE sur les patches seulement ; `key_bias` absent ; layer_norm **sans paramètres**
    ajoutée par Idol après le `norm` du modèle.
12. Décodeur gaussien : découpe **par attribut puis par Gaussienne** ; rotation × 0,1 ; aabb
    `xyz − 0,5` ; `points_offset_perturbation` et `base_offset_scale` lus dans les poids.
13. Octree : enfants dans l'ordre x le plus rapide (i + 2j + 4k) ; compaction ordonnée ; somme
    des comptes conservée à chaque niveau (assertion en debug).
14. PLY : z-up → y-up `(x, −z, y)`, quaternion `(√½, √½, 0, 0) ⊗ q`, clamps avant `logit`/`log`.
15. `QuantizedLinear.weight` est uint32 packé : `computeDType` explicite.
16. Pas de `swift build` pour un binaire (metallib) ; mlx-swift épinglé `exact` ; jamais `branch:`.
17. Jamais de chemin absolu, de poids, de fixture real ni d'en-tête de dépôt gated committé ;
    `parity/*` uniquement par le script Python.
18. Annulation testée **avant** chaque évaluation du débruiteur et entre les niveaux d'octree ; une
    seule synchronisation par pas (IT-2).
