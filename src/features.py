"""Shared settings and input preparation for the PROPHECG-ExECG prediction model.

Every cohort (development, internal validation, external validation, new data) is prepared with the same steps:
  1. blood-pressure changes are z-scored with model/scaler_bp.pkl (transform only; the scaler is never refitted),
  2. missing inputs are set to zero,
  3. the sign of eight latent factors is flipped so that a higher value means a higher predicted risk.
"""
import os

import joblib
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, 'model')
SCALER_BP_PATH = os.path.join(MODEL_DIR, 'scaler_bp.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'revascularization_prediction.json')

# One encoder input per exercise segment: the last median beat of the segment.
# (stage label in the master table, column prefix, latent scaler, VAE checkpoint, latent factors used by the model)
SEGMENTS = [
    ('STAGE 1', 's1', 'scaler_stage1.pkl', 'PROPHECG_ExECG_STAGE1.ckpt',
     [2, 5, 6, 9, 13, 14, 18, 22, 23, 24, 25, 26, 27, 29, 30, 31]),
    ('STAGE 2', 's2', 'scaler_stage2.pkl', 'PROPHECG_ExECG_STAGE2.ckpt',
     [1, 5, 8, 9, 12, 13, 14, 19, 21, 22, 24, 25, 28, 29]),
    ('STAGE 3', 's3', 'scaler_stage3.pkl', 'PROPHECG_ExECG_STAGE3.ckpt',
     [2, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 24, 25, 30]),
    ('STAGE 4', 's4', 'scaler_stage4.pkl', 'PROPHECG_ExECG_STAGE4.ckpt',
     [1, 2, 6, 7, 9, 11, 14, 15, 23, 25, 30, 31]),
    ('#1', 'sr1', 'scaler_stageR1.pkl', 'PROPHECG_ExECG_Recovery.ckpt',     # early recovery
     [2, 6, 11, 12, 14, 15, 17, 18, 19, 21, 23, 24, 25, 26, 28, 29, 32]),
    ('#2', 'sr2', 'scaler_stageR2.pkl', 'PROPHECG_ExECG_Recovery.ckpt',     # mid recovery
     [2, 6, 11, 12, 14, 15, 17, 18, 19, 21, 23, 24, 25, 26, 28, 29, 32]),
]
LATENTS = [f'{prefix}_latent_{i}' for _, prefix, _, _, keep in SEGMENTS for i in keep]

BP_RAW = [f'{prefix}_relative3_{bp}' for _, prefix, _, _, _ in SEGMENTS for bp in ('sys', 'dia')]
BP_SCALED = [c + '_scaled' for c in BP_RAW]
FLIPPED_LATENTS = ['sr2_latent_26', 's4_latent_25', 'sr1_latent_26', 's1_latent_26',
                   'sr2_latent_28', 's3_latent_13', 'sr2_latent_17', 's3_latent_24']


def bp_scaler(dataset, scaler_path=SCALER_BP_PATH):
    """Add the *_scaled blood-pressure columns using the saved development scaler (transform only)."""
    preprocessor = joblib.load(scaler_path)                 # ColumnTransformer('std_scaler': StandardScaler on BP_RAW)
    name, scaler, cols = preprocessor.transformers_[0]      # use the fitted StandardScaler directly so the result does not
    assert list(cols) == BP_RAW                             # depend on the scikit-learn version that reads the pickle
    scaled = scaler.transform(dataset[BP_RAW].values.astype(float))
    dataset = dataset.drop(columns=[c for c in BP_SCALED if c in dataset.columns]).reset_index(drop=True)
    return pd.concat([dataset, pd.DataFrame(scaled, columns=BP_SCALED)], axis=1)


def finalize_features(dataset, scaler_path=SCALER_BP_PATH):
    """Steps 1-3 above, applied to a table of latent factors and raw blood-pressure changes."""
    dataset = bp_scaler(dataset, scaler_path)
    dataset = dataset.fillna(0)
    for c in FLIPPED_LATENTS:
        dataset[c] = -dataset[c]
    return dataset
