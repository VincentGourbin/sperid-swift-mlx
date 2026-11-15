# Piège — `Float.pi` de Swift est arrondi vers zéro

**Symptôme** (2026-11-15, T-1.4, Mac M3 Max) : `PcdPositionEmbedderV2` (angle `x·f·π`, f jusqu'à 1024)
s'écartait de torch de 2,4e-4 en relatif alors que les fréquences étaient identiques au bit et que
`sin` MLX (GPU et CPU) était exact à 1e-7 près sur ses propres angles. 1 297 angles sur 1 500 différaient.

**Cause** : la documentation de Swift le dit — `Float.pi` est « rounded toward zero » :
`0x1.921fb4p1`, alors que `np.float32(np.pi)` et le π fp32 de torch sont arrondis au plus proche,
`0x1.921fb6p1`. Un ulp de π, multiplié jusqu'à ~3000 rad, fait 1 ulp d'angle = 2,4e-4 sur sin/cos.

**Correctif** : `piFloat32 = Float(Double.pi)` (`Sources/IdolCore/Models/Layers/Embedders.swift`),
utilisé partout où la référence multiplie par `torch.pi` ou `np.pi`. Test : `TinyLayersParityTests.positionEmbedderV2`
(2,4e-4 → < 1e-5).

**Effet voisin, non corrigeable** : l'`exp` fp32 de torch (SLEEF) n'est pas correctement arrondi
(13-15 fréquences sur 128 à 1 ulp d'un exp correctement arrondi) ; à 1600 rad cela laisse 1,2e-4 sur
le plongement de niveau. Tolérance dédiée 3e-4 pour la trigonométrie sur grands angles.

**Règle** : dans un port numérique, ne jamais écrire `Float.pi` ; écrire `Float(Double.pi)`. Vaut pour
tous les dépôts Swift/MLX (RoPE, plongements sinusoïdaux, positions 3D).
