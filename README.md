# PROPHECG-ExECG

Pre-trained encoders, prediction model and code for **PROPHECG-ExECG**, an explainable AI model that predicts coronary revascularization within 90 days from exercise electrocardiography (ExECG).

> Manuscript: "Explainable artificial intelligence for prediction of coronary revascularization from exercise electrocardiography" (European Heart Journal – Digital Health, under review).

## Overview

1. **Variational autoencoders (VAEs)**, trained without outcome labels on 12-lead median beats, one per exercise stage (1-4) and one for the recovery phase. Each VAE compresses a median beat into 32 latent factors.
2. **XGBoost classifier** with 102 inputs: 90 selected latent factors (stages 1-4, early and mid recovery) and 12 blood-pressure changes. It outputs the probability of revascularization within 90 days.

Operating thresholds: 0.013 (Youden's index in out-of-fold predictions of the development set, fixed before validation) and 0.068 (matched to physician sensitivity; post hoc).

## Repository structure

```
src/
  preprocess_xml.py              exercise ECG XML -> median beats and master table
  features.py                    segment settings and input preparation
scripts/
  FoundationModel/ecgxaiY/       VAE package (modified from ecgxai), needed to load the encoders
  Downstream/
    predict.py                   latent extraction + prediction
    train_prediction_model.py    XGBoost training and out-of-fold threshold
model/                           scalers and XGBoost model; VAE encoders via GitHub Release (model/README.md)
data/                            not distributed (data/README.md)
```

## Installation

```bash
conda create -n prophecg python=3.8.12
conda activate prophecg
pip install -r requirements.txt
```
Then download the five VAE checkpoints from the GitHub Release into `model/` (see `model/README.md`).

## Usage

```bash
# 1. XML -> median beats and master table
python src/preprocess_xml.py --xml-dirs /path/to/xml_folder --out-dir /path/processed

# 2. prediction
python scripts/Downstream/predict.py --data-dir /path/processed --out predictions.csv

# training of the prediction model (optional)
python scripts/Downstream/train_prediction_model.py --dev development_features.csv --out-dir results/retrained
```

## Data

No patient-level data are included. The ExECG data cannot be shared publicly because of patient privacy; see the Data availability statement of the manuscript.

## Third-party code and license

`scripts/FoundationModel/ecgxaiY` is modified from [UMCUtrecht-ECGxAI/ecgxai](https://github.com/UMCUtrecht-ECGxAI/ecgxai) (van de Leur, Vessies et al.), released under the GNU AGPL-3.0; it also imports base classes from the `ecgxai` package. The modifications (loading of ExECG samples) are marked in the source. This repository is distributed under the **GNU Affero General Public License v3.0** (see `LICENSE`).

## Citation

Boo D, Cho I, Yu HT, Kim M, Park J-W, You SC, Ha J-W. Explainable artificial intelligence for prediction of coronary revascularization from exercise electrocardiography. *European Heart Journal – Digital Health* (under review).
