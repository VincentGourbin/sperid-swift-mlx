## 15. Les autres modèles Speri Labs (préparation, pas de code avant G-6)

| Modèle | Tâche | Taille | Licence | Briques partagées avec Idol |
|---|---|---|---|---|
| `speridlabs/iris-3b` (+ `turbo/`, 4 pas) | texte → image **dans l'espace pixel** (sans VAE), profondeur, restauration | 3 B | Apache-2.0 | DiT flow-matching, échantillonneur Euler/CFG, profils itératifs |
| `speridlabs/iris-z` (small 9,1 B / big 32,2 B) | image + profondeur jointes, image → profondeur, profondeur → image | 9-32 B | FLUX non commerciale | **VAE FLUX.2** (déjà porté au Jalon 3), base FLUX.2-klein-9B (proche de `flux-2-swift-mlx`) |
| `speridlabs/idol-turbo` | Idol en 5 pas | — | Apache-2.0 | Jalon 5 |

Ce qui va dans `SperidCore` quand le deuxième modèle arrive (pas avant : pas d'abstraction sans
deux utilisateurs) : échantillonneur flow-matching générique, `NoiseSource`, profils
`<bits>bit-fast|lean` (type de base), téléchargeur. Iris-Z small est le candidat naturel
(FLUX.2-klein déjà porté dans `flux-2-swift-mlx`, VAE partagé) ; licence non commerciale ⇒ ASK.
