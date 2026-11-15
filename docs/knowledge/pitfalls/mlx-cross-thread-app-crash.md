# Piège — l'app plante dans l'encodeur Metal : graphe MLX construit sur un thread, évalué sur un autre

**Symptôme** (2026-11-15, Idol Studio, première génération de Vincent) : `-[_MTLCommandEncoder dealloc]:
Command encoder released without endEncoding`, pile dans `mlx::core::metal::CommandEncoder::set_input_array`
(table de hachage corrompue) sous `FlowSampler.sample` → `eval`. La CLI n'a jamais planté.

**Cause** : pattern MLX-004 (skill `mlx-swift-patterns`). Le chargement laissait des tableaux paresseux —
poids lus par `loadArrays` sans conversion, experts LoRA castés en fp32 — construits sur le thread de
chargement ; la génération les évaluait sur un autre thread : une `DispatchQueue` série garantit l'ordre des
blocs, **pas le thread** qui les exécute. La CLI fait tout sur un seul thread.

**Correctif** : (1) `IdolPipeline.init` et `IdolDecoder.init` évaluent tous les paramètres, et
`TurboLoRA.materialise` les experts, sur le thread de chargement ; (2) l'app exécute tout appel MLX sur un
**thread unique et durable** (`MLXWorker`), jamais sur une file GCD.

**Règle** : dans une app, un seul thread pour MLX, et rien de paresseux ne sort de lui ; un `[Float]` ou une
valeur évaluée traverse, un `MLXArray` non évalué jamais.
