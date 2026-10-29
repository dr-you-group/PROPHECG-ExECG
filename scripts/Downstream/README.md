# Downstream: revascularization prediction model

## Input preparation (all cohorts)
`src/features.py` holds the segment settings (`SEGMENTS`: latent factors used per stage) and prepares the 102 model inputs in the same way for the development, internal validation, external validation and any new data:

1. The 12 blood-pressure changes (`s1_relative3_sys` ... `sr2_relative3_dia`, stage value minus stage 1 value) are z-scored with `model/scaler_bp.pkl`. The scaler was fitted once on the development set and is only applied (`transform`), never refitted.
2. Missing inputs are set to zero.
3. The sign of eight latent factors (`FLIPPED_LATENTS`) is flipped so that higher values mean higher predicted risk.

## Scripts
| Script | Purpose |
|---|---|
| `predict.py` | Encodes the last median beat of each segment with the pre-trained VAEs, prepares the inputs (`finalize_features`) and predicts with `model/revascularization_prediction.json` |
| `train_prediction_model.py` | Trains the XGBoost model on a development table (latents + raw blood-pressure changes) with the same preparation, derives the operating threshold from out-of-fold predictions (3-fold cross-validation with patient-level folds, Youden's index) and refits the final model |

```bash
python predict.py --data-dir /path/processed --out predictions.csv
python train_prediction_model.py --dev development_features.csv --out-dir ../../results/retrained
```
Add `--latents-already-flipped` if the development table already contains the sign-flipped latent factors.

## Model settings
1,000 trees, maximum depth 10, learning rate 0.01, L2 regularization (lambda) 1, positive-class weight = negatives/positives in the development set, row subsampling 0.7, minimum child weight 9.

A model retrained with `train_prediction_model.py` is not bit-identical to `model/revascularization_prediction.json` (random subsampling), but gave the same out-of-fold AUROC (0.754) and operating threshold (0.013) in the development set.
