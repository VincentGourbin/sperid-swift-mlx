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
