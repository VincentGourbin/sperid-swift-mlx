# Piège — un splat est une coquille, pas un solide (export STL)

**Symptôme** (2026-11-16, renard, premier export STL) : maillage étanche et manifold selon le validateur, mais
70 cm³ pour 155 × 110 × 100 mm, une surface criblée de trous (genre 347) et un crâne ouvert « en bol ». Un
remplissage par inondation depuis l'extérieur, puis une fermeture de 4 mm, ne changeaient rien.

**Causes**, chacune vérifiée sur le nuage :
1. Les Gaussiennes se posent **sur la surface** : la densité seuillée est une coquille, ouverte là où le modèle
   ne voit rien (souvent dessous) ; l'intérieur communique avec l'extérieur et l'inondation le vide.
2. **67 %** des Gaussiennes ont une opacité < 0,05 ; une carte de hauteur vue du dessus montre que le **dessus de
   la tête** n'est fait que d'elles. Le seuil d'opacité du maillage USDZ/GLB (0,05) et son iso (0,04 × p99) l'ôtent.
3. Sur une grille de 250 cellules de haut, la moitié des Gaussiennes dépassent 2 cellules : 7 points
   d'échantillonnage par Gaussienne laissent des cratères.

**Correctif** (`PrintPreparation`) : toutes les Gaussiennes, iso 0,01 × p99 ; intérieur par **vote de 6 rayons**
(plein si ≥ 5 rencontrent de la matière : une face ouverte ne vide plus l'objet, l'espace entre deux pieds reste
vide) ; grille de **128 cellules sur la hauteur**, réglée en cellules et non en mm. Renard : 241 cm³, genre 1,
crâne fermé.

**Règle** : un validateur « étanche + manifold » ne dit rien de la forme ; regarder un rendu (ModelIO + SceneKit
hors écran lit le STL) et le volume avant de conclure. Le maillage visuel (USDZ/GLB) et le solide d'impression
n'ont pas les mêmes réglages.
