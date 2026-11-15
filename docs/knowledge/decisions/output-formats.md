# Décision (proposée) — formats de sortie : PLY, Blender, RealityKit, USD/USDZ

**Contexte** (2026-11-15, question de Vincent : « en sortie on saurait faire usdz et blender ? »).

**Constats vérifiés**
- `.ply` 3DGS (fait, `PLYWriter`, à 1,9e-6 de `save_ply`) : le format d'échange de facto. **Blender 5.3**
  (sortie le 2026-11-10) importe nativement PLY/SPZ de splats et les rend en Workbench, EEVEE et Cycles,
  import seulement (presse : cgchannel, digitalproduction, radiancefields, cinevva, sept. 2026) → Blender
  marche déjà avec nos fichiers, à confirmer à l'œil.
- **RealityKit 27** (macOS/iOS/visionOS 27) : `GaussianSplatResource` + `GaussianSplatComponent`, construits
  depuis des tampons GPU (`LowLevelBuffer` : position, scale, rotation, opacity, SH degré 0-3, activations
  `identity|exponential|sigmoid`, projection, tri). **Vérifié dans le SDK Xcode 27.2** (`RealityFoundation`
  swiftinterface). Pas d'initialiseur depuis un fichier repéré : on alimente directement depuis `GaussianCloud`,
  sans passer par un fichier — idéal pour l'app iOS/Mac.
- **OpenUSD 26.03** : schéma `UsdVolParticleField3DGaussianSplat` (+ convertisseur PLY → USD de référence).
  **AR Quick Look ne rend pas les particle fields** (selon une étude arXiv 2611.10013 ; pas de doc Apple
  contraire trouvée) : un `.usdz` de splats n'est pas un fichier « ouvrable partout » sur iPhone.
- **USDZ universel** (Quick Look, AR, Keynote, Reality Composer) = **maillage** texturé : il faut extraire une
  surface des Gaussiennes (marching cubes sur la densité, ou TSDF depuis des rendus de profondeur), puis
  exporter avec ModelIO (`MDLAsset.export` vers usdz). Même maillage → `.glb`/`.obj` pour Blender en édition.

**Proposition**
1. RealityKit natif (`GaussianCloud` → `GaussianSplatResource`) : peu coûteux, rendu fidèle, cœur de l'app.
2. USD splats (`.usda` + `ParticleField3DGaussianSplat`, empaqueté `.usdz`) : pour les pipelines USD
   (Omniverse, Houdini, Blender futur), pas pour Quick Look.
3. Maillage (`.usdz`, `.glb`, `.obj`) : chantier à part (extraction de surface, cuisson des couleurs), à
   mesurer en qualité ; ce qui rend un objet lisible partout.

**Ouvert** : priorités de Vincent ; Quick Look et le schéma splat (à tester sur iOS 27 en vrai).
