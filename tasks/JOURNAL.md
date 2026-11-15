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
