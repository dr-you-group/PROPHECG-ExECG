"""Predict revascularization within 90 days from preprocessed exercise ECGs.

Input folder (output of src/preprocess_xml.py):
  <data-dir>/whole_validation_master.csv   one row per median beat (PseudoID, AcqDate, TestID, stage, Time,
                                           SampleBase, Gain, relative3_sys, relative3_dia)
  <data-dir>/median/<TestID>.csv           12-lead median beats
Steps
  1. the last median beat of each exercise stage (1-4) and recovery segment (#1 early, #2 mid) is encoded with the
     pre-trained VAE of that segment; the latent means are standardized with the saved latent scalers
  2. finalize_features(): BP scaled with model/scaler_bp.pkl, missing inputs -> 0, latent sign flips
  3. probability of revascularization from model/revascularization_prediction.json (column 'Prob')
ExECGs without a stage 1 median beat are not included (tables are joined on stage 1).

usage:
  python predict.py --data-dir /path/processed --out /path/predictions.csv [--gpus 1]
"""
import argparse
import os
import sys
from functools import reduce

import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'FoundationModel'))
sys.path.append(os.path.join(HERE, '..', '..', 'src'))
from ecgxaiY.systems.VAE_system import GaussianVAE  # noqa: E402
from ecgxaiY.utils.dataset import UniversalECGDataset  # noqa: E402
from ecgxaiY.utils.transforms import ApplyGain, ToTensor  # noqa: E402
from features import MODEL_DIR, MODEL_PATH, SEGMENTS, finalize_features  # noqa: E402


def last_beat_per_segment(master):
    """Keep the last median beat of each stage / recovery segment of every ExECG."""
    return master.groupby(["PseudoID", "AcqDate", "stage"]).apply(lambda x: x.loc[x['Time'].idxmax()]).reset_index(drop=True)


def segment_latents(trainer, vae, beats, waveform_dir, prefix, scaler_path, keep, num_workers=8):
    """Latent factors and blood-pressure changes of one segment, one row per ExECG."""
    dataset = UniversalECGDataset(dataset_function="execg", waveform_dir=waveform_dir, dataset=beats,
                                  transform=transforms.Compose([ApplyGain(), ToTensor()]))
    output = trainer.predict(vae, dataloaders=DataLoader(dataset, batch_size=128, shuffle=False, num_workers=num_workers))
    mu = torch.cat([x["mu"] for x in output], dim=0).cpu().numpy()
    mu = joblib.load(scaler_path).transform(mu)              # development-set scaler of this segment

    latents = pd.DataFrame(mu, columns=[f'{prefix}_latent_{i + 1}' for i in range(mu.shape[1])])
    latents = latents[[f'{prefix}_latent_{i}' for i in keep]]
    bp = beats[['PseudoID', 'AcqDate', 'relative3_sys', 'relative3_dia']].rename(
        columns={'relative3_sys': f'{prefix}_relative3_sys', 'relative3_dia': f'{prefix}_relative3_dia'})
    return pd.concat([bp.reset_index(drop=True), latents], axis=1)


def build_features(master, waveform_dir, trainer, model_dir=MODEL_DIR, num_workers=8):
    """Model inputs (90 latent factors + 12 scaled blood-pressure changes), one row per ExECG."""
    beats = last_beat_per_segment(master)
    vaes, tables = {}, []
    for stage, prefix, scaler, ckpt, keep in SEGMENTS:
        if ckpt not in vaes:                                 # the recovery VAE is shared by #1 and #2
            vaes[ckpt] = GaussianVAE.load_from_checkpoint(checkpoint_path=os.path.join(model_dir, ckpt))
        tables.append(segment_latents(trainer, vaes[ckpt], beats[beats["stage"] == stage], waveform_dir, prefix,
                                      os.path.join(model_dir, scaler), keep, num_workers))
    features = reduce(lambda left, right: left.merge(right, on=["PseudoID", "AcqDate"], how='left'), tables)
    return finalize_features(features)


def predict(features, model_path=MODEL_PATH):
    # The model file was saved from the scikit-learn XGBRegressor wrapper (objective binary:logistic), so
    # XGBClassifier.load_model refuses it; the Booster returns the predicted probability directly.
    booster = xgb.Booster()
    booster.load_model(model_path)
    features['Prob'] = booster.predict(xgb.DMatrix(features[booster.feature_names]))
    return features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True, help='output folder of src/preprocess_xml.py')
    ap.add_argument('--out', required=True, help='output CSV (model inputs and predicted probability)')
    ap.add_argument('--model-dir', default=MODEL_DIR)
    ap.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0)
    ap.add_argument('--num-workers', type=int, default=8)
    a = ap.parse_args()

    trainer = pl.Trainer(logger=False, gpus=a.gpus)
    master = pd.read_csv(os.path.join(a.data_dir, "whole_validation_master.csv"))
    features = build_features(master, os.path.join(a.data_dir, "median"), trainer, a.model_dir, a.num_workers)
    predict(features, os.path.join(a.model_dir, os.path.basename(MODEL_PATH))).to_csv(a.out, index=False)
    print(f'{len(features)} ExECGs -> {a.out}')


if __name__ == '__main__':
    main()
