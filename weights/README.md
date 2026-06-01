# Weights Directory

Place pretrained and fine-tuned model weights here before building the Docker image.

## Required structure

```
weights/
├── manifest.json               ← controls which checkpoints to load and which model config to use
│                                 copy manifest.json.example → manifest.json and edit paths
├── manifest.json.example       ← template (committed, safe to version)
│
├── dinov2_gastronet5m.pth      ← GastroNet-5M pretrained ViT-Base
│                                 Download: https://cortex.thetavision.nl/dataset-provider/listing/2/
│                                 HuggingFace: BONS-AI-TUE-AMC/GastroNetDinov2
│
├── *.pt                        ← fine-tuned ensemble checkpoints (from training)
│                                 Naming convention: {arch}_{ssl}_{fold}_val_ppv_{score}.pt
│                                 e.g. dinov2_fold0_val_ppv_0.9800.pt
│                                      rn50_dino_fold0_val_ppv_0.9600.pt
│
├── patchcore.pkl               ← (optional) PatchCore memory bank — activates hybrid scoring
├── hybrid_config.json          ← (optional) HybridScorer alpha/beta/normalizer
│
└── calibration/
    ├── affine_calibrator.pt    ← AffineCalibrator (preferred — encodes 1% prevalence prior)
    │   OR
    ├── isotonic_calibrator.pkl ← IsotonicCalibrator (fallback)
    └── calibration_results.json ← optimal threshold + bootstrap metrics
```

## manifest.json

`manifest.json` tells `inference.py` which checkpoint maps to which model config.
Without it, all `*.pt` files are loaded as `dinov2_gastronet` (backward compatible).

Format:
```json
[
  {"checkpoint": "dinov2_fold0.pt",  "model_config": "dinov2_gastronet"},
  {"checkpoint": "rn50_dino_fold0.pt", "model_config": "rn50_gastronet"}
]
```

Copy `manifest.json.example` and adjust to your actual checkpoint filenames.

## How to populate

```bash
# 1. Download GastroNet-5M pretrained weights
# 2. Train ViT-B ensemble (5 seeds)
python scripts/train.py project.seed=42
python scripts/train.py project.seed=123
# ... repeat for seeds 456, 789, 1337

# 3. (Optional) Train ResNet50 SSL diversity ensemble
bash scripts/train_ssl_ensemble.sh

# 4. Copy best checkpoints and create manifest.json
cp outputs/seed_42/checkpoints/best.pt weights/dinov2_fold0.pt
# ... edit weights/manifest.json

# 5. Calibrate (affine preferred)
python scripts/calibrate.py calibration=affine
cp outputs/.../results/affine_calibrator.pt   weights/calibration/
cp outputs/.../results/calibration_results.json weights/calibration/

# 6. (Optional) Fit PatchCore
python scripts/fit_patchcore.py \
    --checkpoint weights/dinov2_fold0.pt \
    --model-config configs/model/dinov2_gastronet.yaml \
    --train-csv data/train.csv \
    --val-csv data/val_calibration.csv \
    --out-dir weights/

# 7. Build and test Docker
bash do_test_run.sh
```

## ResNet50 SSL checkpoints

The following GastroNet pretrained ResNet50 checkpoints are supported.
Place them in `weights/` and reference them via `manifest.json`.

| Filename | SSL method | Pretraining data |
|---|---|---|
| `RN50_GastroNet-5M_DINOv1.pth` | DINOv1 | 5M gastro images |
| `RN50_GastroNet-5M_MOCOv2.pth` | MOCOv2 | 5M gastro images |
| `RN50_GastroNet-5M_SIMCLRv2.pth` | SIMCLRv2 | 5M gastro images |
| `RN50_Billion-Scale-SWSL+GastroNet-5M_DINOv1.pth` | SWSL + DINOv1 | 1B web + 5M gastro |

## .gitignore

All `.pt` and `.pkl` files are gitignored — do NOT commit model weights to git.