# Journal d'exécution

Une entrée par fiche, en fin de fichier (gabarit dans `AGENTS.md` §6).

## T-1.0 — Squelette du paquet, scripts, CLAUDE.md, sortie et contrats — 2026-11-15 — validée
- Fait : `Package.swift` (SperidCore, IdolCore, `idol`, SperidTests ; mlx-swift 0.32.3 exact) ; `SperidCore/{SperidError,Loading/{HubFile,ModelsDirectory,SafetensorsHeader,WeightContract}}` ; `IdolCore/{Configuration/{IdolConfig,IdolRequest,IdolWeights},Sampling/{FlowSchedule,NoiseSource},Decoding/SystematicSampler,Output/{GaussianCloud,PLYWriter,CameraPose}}` ; CLI `idol info|schedule` ; `Scripts/{check-build,check-tests,run-tests,link-references,plan-concat}.sh`, `Scripts/reference/dump_headers.py` ; plan rév. 1, docs, base OKF.
- Bug réel rencontré : `SystematicSampler` — la référence divise par `clamp_min(sum, 1)` (une ligne de somme < 1 n'est pas renormalisée) et `1 − 1e-12` vaut 1 en fp32 ; première version fausse, corrigée avant les tests. `searchsorted(side='left')` envoie une valeur sur une frontière dans le bac de gauche (test `[3, 2, 2, 1]`).
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh` → `TESTS OK (19 tests, 0 skippés)`
- Limites connues : aucun modèle ; `GaussianCloud.activate` est la référence CPU de l'activation MLX à venir (T-1.6) ; contrats vérifiés sur les en-têtes publiés seulement (aucun poids local).
- Pas d'agent : 0 · appels d'outils : ~60 (session de conception)
