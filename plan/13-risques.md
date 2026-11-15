## 13. Risques

| # | Risque | Effet | Parade |
|---|---|---|---|
| R-1 | Accès DINOv3 refusé (constaté le 2026-11-15 sur le compte de Vincent) | pas de pipeline de bout en bout | G-0 : demander l'accès, ou repack Comfy (licence DINOv3 à relire) ; Jalons 1-2 n'en dépendent pas |
| R-2 | SDPA non fusionné à L = 12 294 | 4,8 Go de scores par forward, ×2 en CFG | mesure T-2.5 avant tout le reste du Jalon 2 ; requêtes tranchées en `lean` |
| R-3 | `deform_conv2d` (BiRefNet) lent ou faux | Jalon 4 bloqué | protocole `BackgroundRemover`, repli Vision, BiRefNet en dernier |
| R-4 | Divergence de l'octree (partage différent d'une cellule) | ancrages différents, comparaison impossible | bruit rejoué, comptage des cellules divergentes, comparaison finale par rendu et non au bit |
| R-5 | Temps de base sur Mac (≈ 150 s CALCUL) jugé trop long | usage réel en turbo seulement | turbo au Jalon 5 ; profils base gardés comme référence de qualité |
| R-6 | Lanczos non identique à PIL | parité du recadrage approximative | parité pipeline depuis `_input.png`, porte propre au recadrage |
| R-7 | Turbo : qualité inférieure annoncée par ses auteurs (« softer ») | attentes | comparaison visuelle G-5, base reste la référence |
| R-8 | Dérive des dépôts HF (révision changée) | poids différents | `HubFile` épinglé révision + SHA-256, `dump_headers.py` à chaque montée |
