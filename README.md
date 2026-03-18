# mm_rcna_mimic (rebuilt and cleaned)

This package is a cleaned, runnable reconstruction of the original `mm_rcna_mimic` research prototype.
It preserves the original directory layout and high-level pipeline idea:

- MIMIC-CXR study building
- text governance / evidence construction
- multimodal evidence fusion
- retrieval-augmented risk estimation
- conflict mediation
- conformal interval calibration
- repair / abstention loop
- explanation and audit trace

## What changed

The rebuilt version fixes the major structural issues discussed during review:

- verification is stored **per task** instead of keeping only the last task
- retrieval excludes the current study and can also exclude the current subject
- `label_column` is used consistently
- repair actions can trigger real upstream recomputation instead of only editing intervals
- conformal fit writes a JSON artifact that the main pipeline loads automatically
- the package has an explicit README, configs, and batch scripts
- the package can run in a lightweight CPU-only mode without private checkpoints

## Important note

This is still a **research scaffold**. To obtain clinically meaningful results you still need:

1. real patient-level train / calib / test splits
2. independent outcome labels
3. trained vision heads or a stronger image pathway
4. proper retrieval indexes built only from the training split

## Quick start

```bash
python -m pip install -r requirements.txt
python sanity_check.py --config configs/default.yaml
python run_pipeline.py --config configs/default.yaml --study-id demo-study-001
```

## Main scripts

- `build_auto_labels.py`: weak-label generation from tables and report text
- `fit_conformal.py`: fit conformal qhat from a calibration CSV
- `run_pipeline.py`: run the full pipeline for one study
- `run_batch.py`: run the pipeline over many studies
- `run_offline.py`: offline / no-external-tools mode
- `train_contract_model.py`: fit a light verifier model from exported traces
- `train_repair_policy.py`: fit a learned repair policy from exported traces
- `train_vision_heads.py`: minimal trainer for weakly supervised vision heads

## Data layout expected by default config

```text
artifacts/
  index/
    meta.pkl
    image.npy
    text.npy
  labels.csv
  studies.csv
  notes/
    <study_id>.txt
  reports/
    <study_id>.txt
  images/
    <study_id>_0.png
```

You can change all paths in `configs/default.yaml`.
