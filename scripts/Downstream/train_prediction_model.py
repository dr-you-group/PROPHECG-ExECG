"""Train the PROPHECG-ExECG revascularization model on the development set and derive the operating threshold.

Input: a development table with one row per ExECG containing PseudoID, Revascularization (0/1), the 90 latent factors
(s1_latent_*, ..., sr2_latent_*) and the 12 raw blood-pressure changes (s*_relative3_sys/dia), built with the same
pipeline as the validation data (create_validation_dataset in PROPHECG_ExECG_Train.py) but before src/features.py
has been applied. Blood pressure is z-scored with model/scaler_bp.pkl exactly as for the validation cohorts.

Steps
  1. finalize_features(): BP scaled with model/scaler_bp.pkl (transform only), missing inputs -> 0, latent sign flips
  2. out-of-fold predictions by 3-fold cross-validation with folds assigned by patient (stratified by whether the
     patient had an event); the threshold maximizing Youden's index in these predictions is the operating point
  3. the final model is refitted on the whole development set

usage:
  python train_prediction_model.py --dev development_features.csv --out-dir ../../results/retrained
  (add --latents-already-flipped if the table already contains the sign-flipped latent factors)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from features import finalize_features, bp_scaler, MODEL_PATH  # noqa: E402

# final settings of the published model
PARAMS = dict(n_estimators=1000, max_depth=10, learning_rate=0.01, reg_lambda=1.0, subsample=0.7, min_child_weight=9,
              objective='binary:logistic', tree_method='hist', eval_metric='logloss')
SEED = 20261009


def feature_names():
    return json.load(open(MODEL_PATH))['learner']['feature_names']      # the 102 inputs, in model order


def patient_folds(df, k=3, seed=SEED):
    pat = df.groupby('PseudoID')['Revascularization'].max(); rng = np.random.RandomState(seed); fold_of = {}
    for ev in (0, 1):
        ids = pat[pat == ev].index.values.copy(); rng.shuffle(ids)
        for i, pid in enumerate(ids): fold_of[pid] = i % k
    return df['PseudoID'].map(fold_of).values


def fit(X, y, seed=SEED):
    spw = (y == 0).sum() / y.sum()                                      # positive-class weight (negative:positive)
    return xgb.XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=8, **PARAMS).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--latents-already-flipped', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    dev = pd.read_csv(a.dev)
    dev = bp_scaler(dev).fillna(0) if a.latents_already_flipped else finalize_features(dev)
    feats = feature_names(); X = dev[feats].values.astype(float); y = dev['Revascularization'].values.astype(int)

    fold = patient_folds(dev); oof = np.zeros(len(y))
    for k in range(3):
        tr, te = fold != k, fold == k
        oof[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    fpr, tpr, thr = roc_curve(y, oof); j = int(np.argmax(tpr - fpr))
    summary = dict(n=int(len(y)), events=int(y.sum()), oof_auroc=float(roc_auc_score(y, oof)), youden_threshold=float(thr[j]),
                   oof_sensitivity=float(tpr[j]), oof_specificity=float(1 - fpr[j]), params=PARAMS, seed=SEED)

    model = fit(X, y)
    model.save_model(os.path.join(a.out_dir, 'revascularization_prediction_retrained.json'))
    pd.DataFrame({'PseudoID': dev['PseudoID'], 'fold': fold, 'oof_prob': oof}).to_csv(os.path.join(a.out_dir, 'development_oof.csv'), index=False)
    json.dump(summary, open(os.path.join(a.out_dir, 'threshold.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'params'}, indent=1))


if __name__ == '__main__':
    main()
