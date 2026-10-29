# Model files

| File | Content | MD5 |
|---|---|---|
| `PROPHECG_ExECG_STAGE1.ckpt` ... `STAGE4.ckpt` | VAE for the median beats of exercise stages 1-4 (32 latent factors each) | see below |
| `PROPHECG_ExECG_Recovery.ckpt` | VAE for the recovery-phase median beats (used for early (#1) and mid (#2) recovery) | see below |
| `scaler_stage1.pkl` ... `scaler_stage4.pkl`, `scaler_stageR1.pkl`, `scaler_stageR2.pkl` | StandardScaler of the latent means of each segment (fitted on the development set) | |
| `scaler_bp.pkl` | StandardScaler of the 12 blood-pressure changes (fitted on the development set; scikit-learn 1.0.1 ColumnTransformer) | `3e19d6b29b6ae900ad6cf0fb756541c2` |
| `revascularization_prediction.json` | XGBoost classifier (binary:logistic, 1,000 trees, 102 inputs; xgboost 2.0.3) | `f387addb05c87992fe4cdba2c2dfa161` |

## Checkpoints

The five `.ckpt` files (about 346 MB each) exceed GitHub's 100 MB file limit and are attached to the GitHub Release instead of the repository. Download them into this folder:

| File | MD5 |
|---|---|
| `PROPHECG_ExECG_STAGE1.ckpt` | `d9485f0c1a9105ff411df57e2297ce0e` |
| `PROPHECG_ExECG_STAGE2.ckpt` | `630ec5057d05144a5e8279bf313b4f9c` |
| `PROPHECG_ExECG_STAGE3.ckpt` | `aaa053b20f3bdda9c11bc4e6f3b81230` |
| `PROPHECG_ExECG_STAGE4.ckpt` | `ef88fcfeda70f62112fbc95d2b30ad2a` |
| `PROPHECG_ExECG_Recovery.ckpt` | `b7ea928358c05da824fe75230698b3a2` |

The checkpoints were saved on a GPU; load them with `GaussianVAE.load_from_checkpoint(path, map_location='cpu')` to use them without a GPU.
