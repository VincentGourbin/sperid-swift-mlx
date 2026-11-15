# Décision — profils de référence (prévus, rien de mesuré)

**Contexte** (2026-11-15) : appliquer dès la conception le standard YuE2 (`mlx-swift-audit`,
`references/profiles-standard.md` et `iterative-models.md` §4).

**Décision**
- Type `IdolReferenceProfile` (Jalon 6) : `Bits` (4/8/16), `Kind` (fast/lean), `id`, `all`,
  `named(_:)`, `applyGlobalPolicy()` qui sauve et restaure les réglages process-wide.
- Famille sœur `turbo/<bits>bit-<kind>` (même forme) plutôt que des champs sans objet.
- Champs figés (chacun = un bouton existant au moment de l'ajout) : précision par voie (débruiteur,
  décodeurs, DINOv3 ; VAE et BiRefNet non quantifiés), CFG batché ou séquentiel, résidence par étape,
  limites mémoire, pas et CFG du checkpoint (identiques dans une famille), Gaussiennes (262 144).
- Matrice cible et coûts attendus : `docs/References.md`, `plan/03-budget.md`.

**Ouvert** : 4 bits sur l'étape itérée (attendu rejeté, IT-7) ; CFG batché selon T-2.5 ; nombre de
pas d'une éventuelle variante « rapide » de la base (hors profils, après validation visuelle).
