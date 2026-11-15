# parity/

Tier-1 fixtures (weight-free, always green), generated from the **upstream** `idol` Python classes
only — never a reimplementation — by `Scripts/reference/tiny_fixtures.py` (card T-1.3). Do not edit
any file here by hand: the safetensors metadata carries `seed` and `upstream_sha`, and
`Scripts/check-fixtures.sh` regenerates and compares bit for bit. Real-weight fixtures never live
here: they go under `$SPERID_MODELS_DIR/parity/`.
