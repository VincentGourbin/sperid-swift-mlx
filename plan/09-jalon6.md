## 9. Jalon 6 — Profils de référence, quantification, packs, iPhone

| Fiche | Objet | Porte |
|---|---|---|
| T-6.1 | Outil de qualité : rendu de contrôle (PSNR/SSIM à la vue d'entrée + 7 vues) côté Python sur deux `.ply` | métrique reproductible ±0,1 dB |
| T-6.2 | Baseline mesurée `16bit-fast` (base et turbo), protocole `plan/00` | lignes `BENCHMARKS.md` |
| T-6.3 | Quantification par voie (débruiteur 8 bits, décodeurs 8 bits, DINOv3 8 bits ; 4 bits mesuré, attendu rejeté pour l'étape itérée, IT-7) ; couverture en octets (IT-5) | PSNR ≥ 35 dB contre `16bit-fast` |
| T-6.4 | `IdolReferenceProfile` (`<bits>bit-fast\|lean`, `turbo/<bits>bit-*`), `idol references`, `--reference` | chaque champ lu (grep), `docs/References.md` |
| T-6.5 | Packs pré-quantifiés (`int8-mixed`…) + SHA-256 + `Scripts/publish-packs.sh` (n'uploade rien) ⛔ G-6 | Vincent autorise la publication HF |
| T-6.6 | Build iOS, mesure iPhone 15 Pro Max (`8bit-lean` turbo), porte GPU / annulation / checkpoint au pas (IT-12) | pic ≤ 4 Go, ≤ 120 s |
