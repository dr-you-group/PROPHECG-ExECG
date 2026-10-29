# Data

No patient-level data are distributed with this repository. The exercise ECG data cannot be shared publicly because of patient privacy (see the Data availability statement of the manuscript).

Raw exercise ECG XML files are converted with `src/preprocess_xml.py`; its output folder (`whole_validation_master.csv` and `median/`) is the input of `scripts/Downstream/predict.py`. `scripts/Downstream/train_prediction_model.py` expects a development table with one row per ExECG containing `PseudoID`, `Revascularization`, the latent factors and the raw blood-pressure changes.
