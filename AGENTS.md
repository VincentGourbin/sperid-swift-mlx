# AGENTS.md — instructions pour l'agent qui exécute le plan

Tu es l'exécutant du portage **Idol (Speri Labs) → MLX Swift** décrit dans `plan/` (un fichier par
section ; `PLAN.md` n'est qu'un sommaire). Tu travailles **une fiche à la fois**, en français dans
les documents, en anglais dans le code et ses commentaires. Ce fichier est ton mode d'emploi.

## 1. Démarrage d'une session (dans cet ordre, rien d'autre)

1. `read tasks/STATE.md` — la fiche en cours et son état.
2. `read tasks/JOURNAL.md` depuis la fin (les 3 dernières entrées : `grep -n "^## " tasks/JOURNAL.md`).
3. `read tasks/T-<x>.<y>.md` — la fiche en cours (ou la suivante `à faire` de `tasks/INDEX.md` dont
   tous les prérequis sont `validée`).
4. Mets `STATE.md` à jour : `fiche T-x.y · en cours · <date>`.

Une fiche renvoie à **un** fichier `plan/<section>.md`. N'ouvre que celui-là ; jamais deux fichiers
de `plan/` dans le même pas, jamais `Scripts/plan-concat.sh`.

## 2. Pendant une fiche

- Lis **uniquement** ce que la section *À lire* indique, par tranches de **200 lignes** au plus.
  Les sources à copier sont sous `reference/` (`Scripts/link-references.sh` les crée).
- Pour chercher, `grep` un identifiant précis (classe, fonction, clé de tenseur), jamais une phrase.
- Crée les fichiers listés dans *Fichiers à créer*, avec les noms exacts. Un fichier Swift fait
  **200 lignes au plus** ; au-delà, découpe comme la fiche l'indique.
- **Copie d'abord, adapte ensuite** : quand la fiche nomme un fichier source, reprends sa structure
  et ses noms, puis applique les seules modifications listées.
- **Modifie un fichier à la fois** ; ne rédige pas tout le code dans ta réflexion avant d'agir.
- Après chaque fichier écrit : `Scripts/check-build.sh`. Ne continue pas sur un `BUILD FAILED`.
- Lance les validations **dans l'ordre**, arrête-toi au **premier** échec, corrige, relance — et lance-
  les toi-même avant de conclure : une fiche sans sortie de test n'est pas finie.
- Si un test existant contredit le comportement demandé par la fiche, mets-le à jour et dis pourquoi
  dans le journal ; **ne supprime ni ne désactive aucun test**.
- Trois échecs consécutifs sur la même erreur ⇒ question dans `tasks/ASK.md`, `STATE.md` à `bloquée`,
  fin de la réponse.

## 3. Fin d'une fiche

Une fiche est **validée** quand chacune de ses commandes de validation a affiché sa ligne attendue
dans la sortie d'un outil pendant cette session. Alors :

1. Entrée de journal (gabarit ci-dessous) à la **fin** de `tasks/JOURNAL.md`, lignes de validation
   recopiées telles quelles.
2. Fiche à `validée` dans `tasks/INDEX.md` ; `fiche suivante : T-x.y · à faire` dans `tasks/STATE.md`.
3. Fiche **⛔ GATE** : question dans `tasks/ASK.md`, puis **arrêt** (pas de fiche suivante).
4. Termine par `FICHE T-x.y VALIDÉE` (ou `BLOQUÉE`) suivi des lignes de validation.

Tu ne fais **rien** qui ne soit pas dans la fiche : ni refactoring, ni optimisation, ni fichier ou
dépendance en plus. Une amélioration utile va dans `tasks/ASK.md`, pas dans le code.

## 4. Commandes de validation (les seules)

| Commande | Attendu en dernière ligne |
|---|---|
| `Scripts/check-build.sh` | `BUILD OK` |
| `Scripts/check-tests.sh <Suite>` | `TESTS OK (<n> tests, <k> skippés)` |
| `Scripts/check-parity.sh <composant>` | `PARITY OK <composant> …` (à partir de T-1.3) |
| `Scripts/check-fixtures.sh` | `FIXTURES OK` (à partir de T-1.3) |

En cas d'échec, la sortie donne les 20 premières lignes utiles ; le journal complet est sous
`.local-runs/*.log` (`grep -n "error:" .local-runs/build.log`, puis 10 lignes autour). Jamais un log en entier.

## 5. Interdits (les erreurs sont silencieuses)

- `swift build` pour produire un binaire ou lancer les tests (metallib introuvable).
- `branch:` dans `Package.swift` ; changer une version épinglée.
- `eval(model.parameters())` global pendant une conversion de poids.
- Chemin absolu de ta machine dans un fichier committé ; modifier `parity/*` autrement que par le
  script Python ; committer un poids, une fixture real ou un en-tête d'un dépôt gated.
- `git push`, upload, publication ; supprimer un fichier sous `reference/` ou `$SPERID_MODELS_DIR`.
- Générer du bruit avec `MLXRandom` dans un chemin de parité : tout tirage passe par `NoiseSource`.

## 6. Gabarits

**Entrée de journal** (`tasks/JOURNAL.md`) :
```
## T-x.y — <titre> — <AAAA-MM-JJ> — validée|bloquée
- Fait : <fichiers créés/modifiés, une ligne>
- Bug réel rencontré : <un, ou « aucun »>
- Validation : `<commande>` → `<ligne exacte>` (une ligne par validation)
- Limites connues : <ce qui n'est pas couvert, ou « aucune »>
- Pas d'agent : <n> · appels d'outils : <n>
```

**Question à Vincent** (`tasks/ASK.md`) :
```
## ASK — T-x.y — <AAAA-MM-JJ>
- Contexte : <2 lignes>
- Ce que j'ai essayé : <3 lignes max>
- Question : <une question fermée si possible>
- Options : A) … B) …
```

## 7. Repères

- Dépôt : `Sources/SperidCore` (partagé), `Sources/IdolCore` (le modèle), `Sources/IdolCLI` (`idol`),
  `Tests/SperidTests`, `parity/` (fixtures tiny), `Scripts/`, `tasks/`, `reference/` (non committé).
- Poids : `$SPERID_MODELS_DIR/<repo>/<fichier>` (ex. `speridlabs/idol/denoiser.safetensors`). Jamais en dur.
- Faits d'architecture : `plan/02.1-pipeline.md` … `plan/02.8-turbo.md`. Pièges : `plan/11-pieges.md`
  (19 points, cités par numéro dans les fiches).
- Mémoire durable : `docs/knowledge/agent-memory.md` (conventions de Vincent, état, pièges).
