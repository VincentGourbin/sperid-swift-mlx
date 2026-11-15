# Piège — le `.ply` 3DGS d'Idol est y-BAS

**Symptôme** (2026-11-15, Idol Studio, rendu RealityKit hors écran) : les quatre demi-avocats apparaissaient
avec des dessus sombres ; un rendu CPU indépendant (projection de points, Python) donnait la même chose —
l'objet était à l'envers, on voyait la peau sous les fruits.

**Cause** : `Gaussian.save_ply` écrit `(x, y, z)_objet → (x, −z, y)` et documente « the y-up convention
viewers use ». Le haut de l'objet (+z) tombe pourtant sur **−y** : c'est la convention COLMAP / 3DGS d'origine
(y vers le bas), que SuperSplat et la plupart des viewers 3DGS corrigent d'eux-mêmes.

**Correctif** : `SplatEntity` tourne l'entité d'un demi-tour autour de x ; la première vue de contrôle part de la
caméra prédite (`_camera.json`, z-haut → y-haut : (dx, dz, −dy)). Le fichier `.ply` reste identique à celui du
SDK (parité).

**Règle** : un `.ply` 3DGS est y-bas tant que le contraire n'est pas montré par un rendu ; tout moteur y-haut
(RealityKit, Blender en import brut, USD) doit le retourner. À vérifier en T-Blender et T-USD.
