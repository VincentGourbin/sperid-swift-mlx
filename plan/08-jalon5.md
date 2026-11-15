## 8. Jalon 5 — Turbo

| Fiche | Objet | Porte |
|---|---|---|
| T-5.1 | Chargeur LoRA de phase (448 tenseurs) et `LoRALinear` à deux experts | parité real d'un forward slow et fast |
| T-5.2 | Fusion par phase (deux jeux fusionnés) contre delta à la volée : mesure | A/B/B/A, garder la plus rapide si l'écart > 5 %, sinon la plus économe |
| T-5.3 | `IdolRequest.turbo`, `idol generate --turbo` ⛔ G-5 | 8 images, Vincent compare turbo et base |
