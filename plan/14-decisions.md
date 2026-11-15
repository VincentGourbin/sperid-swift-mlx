## 14. Décisions actées et questions ouvertes

### Actées (2026-11-15)
- D-1 **Un paquet, plusieurs cibles** : `SperidCore` (partagé), `IdolCore`, CLI `idol` ; Iris viendra
  en `IrisCore` + CLI `iris` (`plan/15`). Raison : YuE2 (`YuE2Core` + `SheetSage2Core`).
- D-2 **Copie depuis les dépôts frères, pas de dépendance** à `flux-2-swift-mlx` (il tire des
  encodeurs de texte, des chaînes et l'entraînement LoRA : 1 000+ fichiers pour 3 fichiers utiles).
- D-3 **Octree sur CPU**, 8 forwards GPU : formes dépendantes des données.
- D-4 **Bruit injectable partout** (`NoiseSource`) : condition de toute parité.
- D-5 **Ordre des Jalons par vérifiabilité** : décodeurs → débruiteur → encodeurs → prétraitement → turbo → profils.
- D-6 mlx-swift : 0.32.3 exact jusqu'à la v0.3.0 ; **plage `0.31.6..<0.33.0` depuis la v0.3.1** (demande de
  Fluxforge Studio, dont flux-2 et yue2 épinglent 0.31.6 exact : un `exact` ici rendait l'intégration
  irrésoluble). Les deux bornes sont validées sur vrais poids ; `Package.resolved` suit 0.31.6. Profiler ≥ 1.5.1.
- D-7 Dépôt GitHub **public depuis le 2026-11-15** (demande de Vincent, version 0.3.0), audit préalable de tout
  l'historique (aucun chemin absolu, jeton, poids ni image personnelle) ; `THIRD_PARTY_NOTICES.md`.

### Questions (tasks/ASK.md)
- ~~Q-1~~ **Tranchée 2026-11-15 : dépôt officiel, accès demandé.** Source des poids DINOv3 : demander l'accès à `facebook/dinov3-vith16plus-pretrain-lvd1689m`
  (recommandé) ou utiliser `Comfy-Org/Idol` ?
- ~~Q-2~~ **Tranchée 2026-11-15 : nom gardé.** Nom du dépôt et du paquet : `sperid-swift-mlx` / `SperidSwift` / CLI `idol` ?
- ~~Q-3~~ **Tranchée 2026-11-15 : BiRefNet par défaut** (Vision ne trouvait pas le sujet d'une photo de Vincent que le Space détoure), Vision en option. Détourage : BiRefNet porté, ou Vision d'Apple par défaut et BiRefNet en option ?
- Q-4 (après G-4) Un viewer de splats Metal dans ce dépôt, ou dans l'app seulement ?
- Q-6 Formats de sortie : priorités entre RealityKit natif, USD splats et maillage (`.usdz`/`.glb`) — `docs/knowledge/decisions/output-formats.md`.
- Q-5 (G-6) Publication HF des packs (`VincentGOURBIN/idol-mlx-packs`) et visibilité du dépôt.
