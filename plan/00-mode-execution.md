## 0. Mode d'exécution

Ce plan est exécuté fiche par fiche, par Claude Code ou par un exécutant pi (modèle local
`qwen38 serve` ou cloud). Le mode d'emploi de l'exécutant est `AGENTS.md` ; ce fichier fixe la
granularité et les portes.

### 0.1 Granularité (banc `pi-codebench`, skill `pi-plan-decoupage`)

| Exécutant | Une fiche = | Session |
|---|---|---|
| Claude Code, glm-5.3-flash, gemma4 (cloud) | une fonctionnalité cohérente (≤ 4 tâches liées, ≤ 5 fichiers, ≤ 300 lignes) | neuve par fiche |
| Qwen3.8-27B 4 bits (local) | **une tâche** | poursuivie (`pi -p --continue`) tant que le contexte reste sous ~49 k jetons |
| modèle moyen ou non mesuré | une tâche | neuve ; passer d'abord au banc |

Les fiches de `tasks/` sont écrites au grain « une fonctionnalité » ; une fiche marquée
**[découpable]** liste ses sous-tâches pour un exécutant local.

### 0.2 Portes

- `Scripts/check-build.sh` → `BUILD OK` après chaque fichier.
- `Scripts/check-tests.sh <Suite>` → `TESTS OK (n tests, …)`.
- `Scripts/check-parity.sh <composant>` → `PARITY OK <composant> …` (à créer en T-1.3).
- ⛔ **GATE G-n** : la fiche se termine par une question dans `tasks/ASK.md` ; on ne commence
  pas la suivante sans réponse datée de Vincent.

### 0.3 Discipline de mesure (reprise de YuE2, `CLAUDE.md` § Performance)

Aucun chiffre de performance sans `swift-mlx-profiler`, build Release, GPU refroidi, A/B/B/A,
`pgrep` des autres jobs GPU, sorties sous `.local-runs/bench.noindex/`. Un gain non mesuré
s'écrit « attendu ». Un levier sous 5 % est retiré, avec sa mesure dans `docs/knowledge/log.md`.
