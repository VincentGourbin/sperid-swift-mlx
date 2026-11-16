# Piège — un splat n'est pas une densité : la surface est là où la lumière s'éteint (export STL, USDZ)

**Symptômes** (2026-11-16) :
1. Premier STL du renard : étanche et manifold selon le validateur, mais une coquille de 70 cm³ criblée de trous,
   au crâne ouvert. Les Gaussiennes se posent **sur** la surface, et le dessus de la tête n'est fait que de
   Gaussiennes d'opacité < 0,05 (carte de hauteur des centres), que le seuil d'opacité ôtait.
2. Corrigé par toutes les Gaussiennes + iso 0,01 × p99 + vote de 6 rayons : objets fermés mais **bouffis**
   (retour de Vincent, USDZ compris) — une iso basse sur une densité échantillonnée en 7 points puis floutée tombe
   dans les queues des Gaussiennes, 2-3 cellules hors de l'objet.
3. Le vote de 6 rayons (plein si 5 rencontrent de la matière) **remplit les bacs ouverts** : l'aire de jeu du
   baby-foot (parois sur 4 côtés + fond) devenait un plateau plat, barres et joueurs noyés.

**Correctif** (`OpacityField`) : chaque Gaussienne est un noyau d'extinction exact (ellipsoïde à 3σ, élargi à
0,6 cellule en conservant sa profondeur optique), normalisé pour qu'un rayon de face perde α comme dans un rendu
3DGS ; **le solide est ce qu'un observateur au-dessus ou à niveau ne voit pas** (transmittance 0,5, 17 directions).
Les rayons venant de dessous sont ignorés : le modèle ne voit jamais le dessous, souvent ouvert. Les bacs ouverts
vers le haut restent ouverts, l'intérieur reste plein. 60 passes de Taubin effacent les marches d'une cellule
sans faire maigrir (512,9 → 511,9 cm³). Baby-foot : barres et joueurs visibles, pieds fins ; USDZ 29 → 13 Mo.

**Règles** : un validateur « étanche + manifold » ne dit rien de la forme — rendre (ModelIO + SceneKit pour le
STL, `idol-studio --render` pour l'USDZ) et comparer au rendu du splat lui-même. Une iso sur une densité floutée
place la surface dans les queues ; la surface d'un splat se définit par la transmittance.

**Piège annexe** : `withUnsafeMutableBufferPointer` sur un tableau **vide** donne une adresse non nulle (le stockage
vide partagé, en lecture seule) — y écrire = SIGBUS (`OpacityField.extinction` sans couleur).
