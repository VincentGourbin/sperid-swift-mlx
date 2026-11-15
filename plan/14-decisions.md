## 14. Décisions actées et questions ouvertes

### Actées (2026-11-15)
- D-1 **Un paquet, plusieurs cibles** : `SperidCore` (partagé), `IdolCore`, CLI `idol` ; Iris viendra
  en `IrisCore` + CLI `iris` (`plan/15`). Raison : YuE2 (`YuE2Core` + `SheetSage2Core`).
- D-2 **Copie depuis les dépôts frères, pas de dépendance** à `flux-2-swift-mlx` (il tire des
  encodeurs de texte, des chaînes et l'entraînement LoRA : 1 000+ fichiers pour 3 fichiers utiles).
- D-3 **Octree sur CPU**, 8 forwards GPU : formes dépendantes des données.
- D-4 **Bruit injectable partout** (`NoiseSource`) : condition de toute parité.
- D-5 **Ordre des Jalons par vérifiabilité** : décodeurs → débruiteur → encodeurs → prétraitement → turbo → profils.
- D-6 mlx-swift **0.32.3 exact** (dernière montée d'un dépôt frère, gemma-4-swift-mlx) ; profiler ≥ 1.5.1.
- D-7 Dépôt GitHub **privé** jusqu'à décision de Vincent ; règles d'un dépôt public appliquées dès
  maintenant (aucun chemin absolu, aucun poids).

### Questions (tasks/ASK.md)
- Q-1 (G-0) Source des poids DINOv3 : demander l'accès à `facebook/dinov3-vith16plus-pretrain-lvd1689m`
  (recommandé) ou utiliser `Comfy-Org/Idol` ?
- Q-2 (G-0) Nom du dépôt et du paquet : `sperid-swift-mlx` / `SperidSwift` / CLI `idol` ?
- Q-3 (Jalon 4) Détourage : BiRefNet porté, ou Vision d'Apple par défaut et BiRefNet en option ?
- Q-4 (après G-4) Un viewer de splats Metal dans ce dépôt, ou dans l'app seulement ?
- Q-5 (G-6) Publication HF des packs (`VincentGOURBIN/idol-mlx-packs`) et visibilité du dépôt.
