# Journal d'exécution

Une entrée par fiche, en fin de fichier (gabarit dans `AGENTS.md` §6).

## T-1.0 — Squelette du paquet, scripts, CLAUDE.md, sortie et contrats — 2026-11-15 — validée
- Fait : `Package.swift` (SperidCore, IdolCore, `idol`, SperidTests ; mlx-swift 0.32.3 exact) ; `SperidCore/{SperidError,Loading/{HubFile,ModelsDirectory,SafetensorsHeader,WeightContract}}` ; `IdolCore/{Configuration/{IdolConfig,IdolRequest,IdolWeights},Sampling/{FlowSchedule,NoiseSource},Decoding/SystematicSampler,Output/{GaussianCloud,PLYWriter,CameraPose}}` ; CLI `idol info|schedule` ; `Scripts/{check-build,check-tests,run-tests,link-references,plan-concat}.sh`, `Scripts/reference/dump_headers.py` ; plan rév. 1, docs, base OKF.
- Bug réel rencontré : `SystematicSampler` — la référence divise par `clamp_min(sum, 1)` (une ligne de somme < 1 n'est pas renormalisée) et `1 − 1e-12` vaut 1 en fp32 ; première version fausse, corrigée avant les tests. `searchsorted(side='left')` envoie une valeur sur une frontière dans le bac de gauche (test `[3, 2, 2, 1]`).
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh` → `TESTS OK (19 tests, 0 skippés)`
- Limites connues : aucun modèle ; `GaussianCloud.activate` est la référence CPU de l'activation MLX à venir (T-1.6) ; contrats vérifiés sur les en-têtes publiés seulement (aucun poids local).
- Pas d'agent : 0 · appels d'outils : ~60 (session de conception)

## T-1.1 — Environnement Python de référence — 2026-11-15 — validée
- Fait : `Scripts/reference/setup_venv.sh` (uv, Python 3.12, idol-sdk editable, torch 2.15.0, MPS disponible), `Scripts/reference/run_sdk.py` (une image → `.ply`, `_input.png`, `_camera.json`).
- Bug réel rencontré : aucun.
- Validation : `.venv-ref/bin/python -I -c "import idol; print('IDOL OK')"` → `IDOL OK`
- Limites connues : passe SDK complète reportée à l'accès DINOv3 (`auth_check` → `GatedRepoError` le 2026-11-15) ; FLUX.2-dev accessible. Le Jalon 1 n'en dépend pas.
- Pas d'agent : 0 · appels d'outils : 4

## T-1.2 — Téléchargement des poids et `idol download` — 2026-11-15 — validée
- Fait : `SperidCore/Loading/{HubDownloader,ChunkedDownloadDelegate}.swift` (copie adaptée de YuE2 `ModelDownloader` : `[HubFile]`, `.part` + Range, SHA-256 épinglé, fichier présent et vérifié sauté, 401/403 ⇒ `gatedRepository`), `IdolCLI/DownloadCommand.swift` (`idol download [--turbo]`), CLI passé en `AsyncParsableCommand` ; `Tests/SperidTests/HubDownloaderTests.swift` (serveur `URLProtocol` factice).
- Bug réel rencontré : aucun.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh HubDownloaderTests` → `TESTS OK (4 tests, 0 skippés)` ; `idol info` → `ok` sur denoiser, decoder, decoder_gs, birefnet (et idol_turbo_lora)
- Limites connues : durée non relevée (quelques minutes, 3,2 Go) ; DINOv3 et VAE pas encore téléchargeables par `idol download` (Jalon 3).
- Pas d'agent : 0 · appels d'outils : 5

## T-1.3 — Fixtures tiny, `check-fixtures.sh`, `check-parity.sh` — 2026-11-15 — validée
- Fait : `Scripts/reference/{_common,tiny_fixtures}.py` (classes amont, C 64, 4 têtes × 16, ctx 16, fp32 forcé via `module.dtype`), `parity/tiny_layers.safetensors` (193 tenseurs : normes, RMSNorm, FFN, pos v2, plongements temps/niveau, attention self/cross, deux blocs, décodeur octree + descente complète 256 points / 4 niveaux, décodeur gaussien + `to_representation`), `Scripts/check-fixtures.sh`, `Scripts/check-parity.sh`.
- Bug réel rencontré : les décodeurs amont forcent fp16 par `self.dtype` (et `convert_module_to_f16`) même sur CPU — remis en fp32 pour le tier 1. Tirages de `systematic_sample` groupés par `counts.unique(sorted=False)` : l'enregistreur recalcule les groupes et sert `torch.rand` depuis une table par cellule (fonction amont inchangée), d'où des offsets rejouables dans l'ordre des cellules.
- Validation : `.venv-ref/bin/python -I Scripts/reference/tiny_fixtures.py` → `wrote …/parity/tiny_layers.safetensors (193 tensors, 2168870 bytes)` ; `Scripts/check-fixtures.sh` → `FIXTURES OK`
- Limites connues : fichier 2,17 Mo (seuil indicatif de la fiche 2 Mo, accepté) ; descente tiny à 4 niveaux (real : 8).
- Pas d'agent : 0 · appels d'outils : 4

## T-1.4 — Couches communes Swift et parité tiny — 2026-11-15 — validée
- Fait : `Sources/IdolCore/Models/Layers/{Norms,Embedders,Attention,CrossBlocks,ModuleLoading}.swift` (LayerNorm32 à eps paramétré, MultiHeadRMSNorm L2, SinusoidalEmbedder, PcdPositionEmbedderV2, FeedForwardNet GELU tanh, MultiHeadAttention self/cross, ModulatedCrossOnlyBlock, TransformerCrossBlock, chargement strict `verify: .all`) ; `Tests/SperidTests/{TinyParity,TinyLayersParityTests}.swift`.
- Bug réel rencontré : `Float.pi` de Swift arrondi vers zéro (piège 20, `docs/knowledge/pitfalls/swift-float-pi-rounded-toward-zero.md`) ; `args * 2 * π` en deux arrondis ; fréquences v2 : `torch.linspace` = FMA par moitié ; générateur : `randomize` écrasait les buffers non persistants (corrigé, fixtures régénérées).
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh TinyLayersParityTests` → `TESTS OK (7 tests, 0 skippés)` (rel < 1e-5, trig grands angles < 3e-4) ; `Scripts/check-fixtures.sh` → `FIXTURES OK`
- Limites connues : tolérance trig 3e-4 (l'`exp` fp32 de torch n'est pas correctement arrondi).
- Pas d'agent : 0 · appels d'outils : 12

## T-1.5 — Décodeur octree et descente (bruit rejoué) — 2026-11-15 — validée
- Fait : `Models/Decoders/OctreeDecoder.swift`, `Decoding/OctreeDescent.swift` (CPU, probabilités par closure), `Sampling/ReplayNoise.swift` (+ `SeededNoise`), `Models/Layers/SinusoidalTables.swift` (généré par `Scripts/reference/dump_sinusoid_tables.py`), `ModuleLoading.referencePrecision` (Linear des blocs en fp16, normes et reste en fp32), `Scripts/reference/real_fixtures.py` (latent réel par la branche **non conditionnée** du vrai débruiteur, sans encodeur ; descente fp16 sur MPS + descente fp32 CPU, tirages et cellules par niveau enregistrés), `Tests/SperidTests/{TinyOctreeParityTests,RealDecoderParityTests}.swift`.
- Bug réel rencontré : (1) en fp16 une différence de probabilité de 1e-4 fait passer un point dans la cellule voisine dès le niveau 2 (49 cellules au lieu de 48) : la comparaison au bit n'existe qu'en fp32, le fp16 se juge par recouvrement d'occupation ; (2) **fixture** : les constructeurs amont passent les blocs en fp16 avant `load_state_dict`, un `.float()` après coup donnait une « référence fp32 » aux poids arrondis fp16 (écart 3e-3 → 4e-5 une fois corrigé) ; (3) `ReplayNoise` ne tue plus le process sur un compte différent, il le signale.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh TinyOctreeParityTests` → `TESTS OK (3 tests, 0 skippés)` ; `Scripts/check-tests.sh RealDecoderParityTests` → `TESTS OK (3 tests, 0 skippés)` avec `REALOCTREE fp32 cells [8, 49, 213, 910, 3489, 7336, 8132, 8181] exact true leafOverlap 1.0 mismatches []`, `fp32 max |Δp| ≤ 3.9e-5`, `fp16 max |Δp| ≤ 5.3e-3`, `fp16 overlap16 0.978 overlap32 0.907`
- Limites connues : la porte « au bit » vaut en fp32 ; en fp16 (précision de référence) porte statistique (recouvrement 16³ ≥ 0,97). Débruiteur PyTorch MPS : 358 s pour 50 évaluations (7,2 s chacune), repère non MLX.
- Pas d'agent : 0 · appels d'outils : 12

## T-1.6 — Décodeur gaussien et `to_representation` — 2026-11-15 — validée
- Fait : `Models/Decoders/GaussianDecoder.swift` (16 blocs, buffers `points_offset_perturbation`/`base_offset_scale` chargés), `Decoding/GaussianAssembly.swift` (découpe par attribut, offsets, `lr`, activations en MLX, constantes de biais en fp32 comme torch), `GaussianCloud.activate` aligné sur les mêmes constantes ; `Tests/SperidTests/TinyGaussianParityTests.swift` ; trois tests real ; fixture fp32 du décodeur gaussien (CPU, 97 s).
- Bug réel rencontré : aucun dans le port ; **porte de la fiche mal calibrée** — rel L2 ≤ 1e-3 en fp16 est sous le bruit du fp16 de la référence elle-même (Python MPS fp16 à 1,8e-2 de son fp32). Portes recalibrées (voir la fiche).
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh TinyGaussianParityTests` → `TESTS OK (1 tests, 0 skippés)` ; `Scripts/check-tests.sh RealDecoderParityTests` → `TESTS OK (6 tests, 0 skippés)` avec `REALGAUSS fp32 relL2 4.96964e-06`, `REALGAUSS fp16 vs fp32: swift 0.003381391 python 0.01822612`, `REALGAUSS assembly worst relMax 1.1920964e-07`
- Limites connues : temps du forward à 8 192 / 32 768 ancrages non profilé (à faire avec le protocole de mesure, pas en test) ; 1 M de Gaussiennes non essayé.
- Pas d'agent : 0 · appels d'outils : 6

## T-1.7 — `idol decode` : latent → `.ply` ⛔ G-1 — 2026-11-15 — validée (GATE G-1 posée)
- Fait : `Decoding/IdolDecoder.swift` (les deux décodeurs chargés après vérification des contrats, `decode(latent:numGaussians:noise:)`), `IdolCLI/{DecodeCommand,ParityCommand}.swift` (`idol decode`, `idol parity decode`), `Scripts/reference/compare_ply.py`, `Scripts/check-parity.sh` (étape PLY côté Python pour `decode`).
- Bug réel rencontré : aucun.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-parity.sh decode` → `PARITY OK decode.anchors.fp32 exact=true cells=[8, 49, 213, 910, 3489, 7336, 8132, 8181]`, `PARITY OK decode.gaussian.fp32 relL2=4.96964e-06`, `PARITY OK decode (fp32 anchors exact, Gaussian fp32 relL2 4.96964e-06)`, `PARITY OK decode.ply PLY OK 262144 vertices, max |Δ| 1.91e-06 (tol 1e-05)` ; `idol decode` → `DECODE OK 262144 Gaussians … 10.1 s` (chargement compris, pas une mesure)
- Limites connues : le latent vient de la branche non conditionnée (pas d'image) ; la passe SDK complète sur l'avocat (DINOv3 désormais accessible) tourne encore et n'enregistre pas son latent.
- Pas d'agent : 0 · appels d'outils : 5

## T-2.1 · T-2.2 · T-2.3 — Couches du débruiteur (plongements v1, RePo + rotation entrelacée, attention RoPE, bloc unifié) — 2026-11-15 — validées
- Fait : `Models/Layers/Embedders.swift` (`PcdPositionEmbedderV1`), `Models/Denoiser/{RePo,UnifiedBlock,Denoiser,DenoiserLoading}.swift` (`RePo3DRotaryEmbedding` avec `clamp_mul` écrit comme la référence, `applyInterleavedRotary`, `RopeMultiHeadAttention` RoPE **avant** les normes, `UnifiedTransformerBlock` modulé / non modulé, `IdolDenoiser` complet, `FeedForwardStack`) ; `referencePrecision` restreint aux poids/biais des Linear (les `shift_table` des blocs restent fp32) ; fixture `parity/tiny_denoiser.safetensors` (211 tenseurs, modèle tiny C 128 × 2 têtes × 64) ; `Tests/SperidTests/TinyDenoiserParityTests.swift`.
- Bug réel rencontré : amont — `MLP(mlp_layer_num=1)` est mal formé (`Linear(inner, out)` sur une entrée de 5 canaux) ; le vrai modèle a 2 raffineurs, la fixture aussi.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh TinyDenoiserParityTests` → `TESTS OK (6 tests, 0 skippés)` (`TINYDIT latent 6.177498e-07 camera 5.804212e-07`) ; `Scripts/check-fixtures.sh` → `FIXTURES OK`
- Limites connues : chemin complexe (`real_valued = False`, MPS) non porté — la forme réelle (cos, sin) est la même rotation.
- Pas d'agent : 0 · appels d'outils : 6

## T-2.4 — `LatentSeqMMFlowModel` réel + chargeur (560 tenseurs) — 2026-11-15 — validée
- Fait : `IdolDenoiser.load` (contrat vérifié, Linear des `blocks`/`noise_refiner`/`context_refiner` en fp16, reste fp32) ; `Scripts/reference/denoiser_fixtures.py` (**objet réel** : avocat de la galerie, prétraitement + DINOv3 + VAE du SDK, bruit graine 0, forward t = 1000 en fp16 MPS et fp32 CPU sur les deux branches, trajectoire 50 pas CFG 3 avec points de contrôle par hook, `.ply` décodé par le SDK) ; `Tests/SperidTests/RealDenoiserParityTests.swift`.
- Bug réel rencontré : aucun dans le port. Seuil fp32 initial (1e-4) trop serré : 30 blocs sur 12 294 jetons, ordre de sommation CPU torch ≠ GPU MLX → 7,7e-5 (cond) et 2,6e-4 (neg) ; seuil 5e-4, quatre fois sous le bruit fp16.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh RealDenoiserParityTests` → `TESTS OK (2 tests, 0 skippés)` avec `REALDIT fp32 cond latent 7.684536e-05 camera 1.1688283e-05`, `REALDIT fp32 neg latent 0.00025806317 camera 2.7549675e-05`, `REALDIT fp16 cond vs fp32: swift 0.0011472855 python 0.0011109156`, `REALDIT fp16 neg vs fp32: swift 0.0010723813 python 0.0011968968`
- Limites connues : référence fp16 sur MPS (pas CUDA) ; une seule image.
- Pas d'agent : 0 · appels d'outils : 6

## T-2.6 — Échantillonneur Euler + CFG — 2026-11-15 — validée
- Fait : `Sampling/FlowSampler.swift` (`IdolCondition` + `negative` = zéros avant projections, CFG séquentiel ou batché, annulation avant chaque pas, une synchronisation par pas, observateur de pas) ; `idol parity denoise` (`IdolCLI/ParityDenoise.swift`).
- Bug réel rencontré : aucun.
- Validation : `Scripts/check-parity.sh denoise` → `PARITY OK denoise.final latent relL2=0.0058743255 camera relL2=0.0010423532` ; trajectoire `step 10 relL2=0.00014134933`, `step 25 relL2=0.0005064725`, `step 49 relL2=0.0055004708` ; caméra `swift yaw 1.410 pitch 0.355 fov 0.12 | python yaw 1.409 pitch 0.355 fov 0.12` ; `PARITY OK denoise (final latent relL2 0.0058743255)`
- Limites connues : CFG batché non encore comparé (même formule ; à décider en T-2.5) ; 491 s pour 100 évaluations dans la parité (pas une mesure : voir T-2.5).
- Pas d'agent : 0 · appels d'outils : 2

## T-2.5 — Mesure : un forward L = 12 294, lot 1 et 2 — 2026-11-15 — validée (décision mémoire), temps à remesurer
- Fait : `idol profile denoiser [--batch 1|2] [--runs N]` (`IdolCLI/ProfileCommand.swift`, swift-mlx-profiler, entrées aléatoires aux formes réelles, 1 forward d'échauffement) ; A/B/B/A avec 90 s de refroidissement, GPU libre (pgrep).
- Mesure : lot 1 = 5,76 s (A1) puis 3,36 s (A2) ; lot 2 = 7,21 et 7,03 s, soit 3,61 et 3,52 s par échantillon. Pic MLX 2,06 Go (lot 1) / 2,75 Go (lot 2).
- **Décision (mémoire, robuste)** : l'attention n'est **pas** matérialisée — le lot 2 coûte +0,7 Go, pas les +4,8 Go de scores attendus s'ils l'étaient (`plan/03-budget.md` §3.3). Le CFG batché tient sans risque mémoire, même en profil économe.
- **Temps (fragile)** : A1/A2 diffèrent de 70 %, hors de la dispersion ±2 % du protocole — état des horloges GPU. Le meilleur point (3,36 s ≈ 6,5 TFLOPS effectifs sur 22 TFLOP) est sous la fourchette CALCUL (7-15 TFLOPS) ; batché ≈ séquentiel par échantillon, gain non démontré. Remesure longue (plus de runs, échantillon de contrôle) avant toute conclusion de vitesse.
- Validation : `idol profile denoiser --batch 1|2 --runs 3` → `PROFILE denoiser batch=1 peak_mlx=2.16 GB`, `PROFILE denoiser batch=2 peak_mlx=2.89 GB` (×2 chacun) ; lignes ajoutées à `BENCHMARKS.md`
- Repère : PyTorch MPS 7,2 s par forward (même machine, T-1.5) → MLX 2,1× plus rapide au meilleur point.
- Pas d'agent : 0 · appels d'outils : 3

## T-2.7 — `idol denoise` → latent ; chaîne avec le décodage → `.ply` ⛔ G-2 — 2026-11-15 — validée (GATE G-2 posée)
- Fait : `IdolCLI/DenoiseCommand.swift` (`idol denoise --cond … --out latent.safetensors [--ply x.ply] [--noise …] [--guidance sequential|batched]`) ; `idol parity denoise` écrit l'avocat décodé tout en Swift.
- Bug réel rencontré : aucun.
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-parity.sh denoise` → `PARITY OK denoise (final latent relL2 0.0058743255)` ; `.local-runs/parity/avocado_swift.ply` (262 144 Gaussiennes)
- Limites connues : le conditionnement vient encore de Python (DINOv3 + VAE : Jalon 3).
- Pas d'agent : 0 · appels d'outils : 2

## T-3.1 → T-3.5 — Encodeurs d'image en Swift — 2026-11-15 — validées
- Fait : `Models/Encoders/{DINOv3,FluxVAEEncoder,FluxVAETokens,EncoderWeights}.swift` (DINOv3 ViT-H+/16 : patch conv, CLS + 4 registres, RoPE 2D `rotate_half` sur les patches, layerscale, gated SiLU, layer_norm finale d'Idol ; encodeur VAE FLUX.2 aux clés diffusers, GroupNorm à statistiques fp32, posterior échantillonné avec ε rejoué, patchification 2×2, BN eps 1e-4, 5 lignes de zéros) ; poids lus dans le dossier des modèles puis le cache HF (dépôts officiels gated) ; `Scripts/reference/encoder_fixtures.py` ; `Tests/SperidTests/RealEncoderParityTests.swift`.
- Bug réel rencontré : fixture — `vae32_mean` et `vae32_feature2` n'avaient pas la même entrée (le SDK arrondit et calcule `·2 − 1` en bf16) ; Swift : `moments(sdkInput:)`. GroupNorm MLX en bf16 calcule ses statistiques en bf16 (torch : fp32) → `GroupNorm32`.
- Validation : `Scripts/check-tests.sh RealEncoderParityTests` → `TESTS OK (2 tests, 0 skippés)` avec `REALENC dinov3 float32 vs fp32: relL2 1.785392e-06 cos 1.0`, `REALENC dinov3 bfloat16 vs fp32: relL2 0.011745531 cos 0.99993104 | python bf16 relL2 0.011218815`, `REALENC vae float32 vs fp32: mean 2.070674e-05 logvar 2.0443846e-05 feature2 1.608539e-05`
- Limites connues : VAE en fp32 par défaut (bf16 Swift plus loin du fp32 que le bf16 du SDK) ; DINOv3 en bf16 comme la référence.
- Pas d'agent : 0 · appels d'outils : ~15

## T-4.1, T-4.2, T-4.6 — Prétraitement, détourage Vision, pipeline public ⛔ G-4 — 2026-11-15 — validées (BiRefNet T-4.3-4.5 reportées)
- Fait : `Preprocess/{RGBAImage,BackgroundRemover,ImagePreprocessor}.swift` (côté court, alpha réel ou `VisionBackgroundRemover`, érosion 3×3, carré × marge, Lanczos vImage prémultiplié, composite noir), `Pipeline/IdolPipeline.swift` (`generate(image:request:noise:…)`, progression, annulation, `IdolOutput.write` = mêmes trois fichiers que le SDK), `idol generate` ; `Output/PLYReader.swift` (+ test aller-retour) ; app `Sources/IdolStudio` (réglages et exemples du Space, viewer RealityKit 27, ouverture de `.ply`, rendu hors écran `--render`).
- Bug réel rencontré : convention y-bas du PLY (piège 21) ; `pgrep -f` qui se trouve lui-même dans une boucle d'attente (deux fois).
- Validation : `idol generate --image …/avocado.webp` → `GENERATE OK avocado: 262144 Gaussians — preprocess 0.1 s, encode 3.3 s, denoise 596.0 s, decode 3.1 s` ; image prétraitée vs SDK : écart moyen 0,35/255, IoU 0,984 ; `TESTS OK (1 tests)` PLYRoundTrip ; rendu `RENDER OK` (vue de la caméra prédite = image d'entrée)
- Limites connues : détourage Vision ≠ BiRefNet (images sans alpha) ; IoU 0,984 < 0,99 (bords Lanczos) ; débruitage 10 min (turbo : Jalon 5).
- Pas d'agent : 0 · appels d'outils : ~25

## T-5.1 → T-5.3 — idol-turbo — 2026-11-15 — validées (G-5 : revue visuelle envoyée)
- Fait : `Models/Denoiser/TurboLoRA.swift` (`PhaseLoRALinear` : base + delta fp32 `(x·Aᵀ·Bᵀ)·4`, deux experts commutés par `TurboPhaseState` ; injection dans les 28 blocs via `update(modules:)`), `FlowSampler(turbo:)` (slow si t ≥ 0,5, fast sinon), `IdolPipeline(useTurbo:)` (LoRA injectée si présente, phase nulle = base exacte), `idol generate --turbo`, bascule Turbo dans Idol Studio (par défaut) ; `Scripts/reference/turbo_fixtures.py` (le `turbo.py` officiel) ; `Tests/SperidTests/RealTurboParityTests.swift`.
- Bug réel rencontré : aucun. T-5.2 (fusion par phase contre delta à la volée) : delta à la volée retenu sans mesure comparative (fidèle à la référence, +3 % FLOPs CALCUL) — à mesurer avec les leviers de performance.
- Validation : `Scripts/check-tests.sh RealTurboParityTests` → `TESTS OK (3 tests, 0 skippés)` avec `REALTURBO float32 slow vs fp32: swift 0.00014110006`, `fast 0.00017150278`, `float16 slow 0.0018322907 (python 0.0016136391)`, `fast 0.0036271727 (python 0.0035646113)`, `5-step final latent relL2 0.002539` ; `idol generate --turbo` sur 4 exemples du Space → 18,8-24,7 s de débruitage, ≈ 25-31 s au total par objet.
- Limites connues : temps non mesurés selon le protocole (ordre de grandeur).
- Pas d'agent : 0 · appels d'outils : ~10
