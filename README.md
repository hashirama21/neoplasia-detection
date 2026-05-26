# RARE26 — Barrett Neoplasia Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/neoplasia-detection/blob/main/RARE2026.ipynb)

PPV@90Recall optimized pipeline for the [RARE26 Grand Challenge](https://grand-challenge.org/challenges/rare26/).

## Architecture

```
GastroNet-5M DINOv2 ViT-Base (336×336)
    + LoRA fine-tuning (rank 8)
    + Lightweight head: Linear(768→256→1)
    + AsymmetricLoss (gamma_neg CV-tuned)
    + Post-training: Isotonic Regression calibration
    + Threshold: optimized via 1000-bootstrap simulation at 1% prevalence
    + Inference: TTA (8 views) × Ensemble (5 models)
```

## Project Structure

```
rare26/
├── configs/                  # Hydra configs
│   ├── config.yaml           # Main config
│   ├── model/dinov2_gastronet.yaml
│   ├── data/rare26.yaml
│   ├── training/base.yaml
│   ├── calibration/isotonic.yaml
│   └── inference/tta_ensemble.yaml
├── src/
│   ├── models/rare26_model.py      # ViT-Base + LoRA + lightweight head
│   ├── losses/asymmetric_loss.py   # ASL + Focal Loss (ablation)
│   ├── data/
│   │   ├── dataset.py              # Dataset + endoscopy augmentations
│   │   └── datamodule.py           # Strict val_selection/val_calibration split
│   ├── training/trainer.py         # Trainer + CV gamma sweep + checkpointing
│   ├── calibration/calibrator.py   # Isotonic Regression + threshold search
│   ├── inference/predictor.py      # TTA + Ensemble predictor
│   └── utils/metrics.py            # PPV@90Recall + bootstrap simulation
├── scripts/
│   ├── train.py                    # Main training entry point (Hydra)
│   ├── calibrate.py                # Standalone calibration
│   └── evaluate.py                 # Local evaluation with bootstrap
├── tests/test_metrics.py           # Unit tests
├── inference.py                    # Docker entry point
├── docker/Dockerfile               # Grand Challenge submission image
├── do_test_run.sh                  # Test Docker locally
├── do_save.sh                      # Export .tar.gz for upload
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

**Download GastroNet-5M pretrained weights:**
```bash
mkdir -p weights
# From HuggingFace: BONS-AI-TUE-AMC/GastroNetDinov2
# Or: https://cortex.thetavision.nl/dataset-provider/listing/2/
wget -O weights/dinov2_gastronet5m.pth <checkpoint_url>
```

**Prepare data:**
```
data/
├── train.csv              # columns: image_path, label (0/1)
├── val.csv                # official validation set
├── val_selection.csv      # auto-generated: 70% of val for model selection
└── val_calibration.csv    # auto-generated: 30% of val for threshold calibration ONLY
```

## Training

```bash
# Standard training (gamma_neg from CV)
python scripts/train.py

# Override gamma_neg directly (skip CV)
python scripts/train.py training.cross_validation.enabled=false training.loss.gamma_neg=4

# Hyperparameter sweep with Hydra multirun
python scripts/train.py --multirun training.loss.gamma_neg=2,4,6

# Full experiment with custom name
python scripts/train.py project.experiment_name=lora_rank8_gamma4
```

## Calibration

```bash
# After training — run on val_calibration ONLY
python scripts/calibrate.py paths.checkpoint_dir=outputs/.../checkpoints

# Compare methods
python scripts/calibrate.py calibration.method=isotonic
python scripts/calibrate.py calibration.method=temperature
```

## Evaluation

```bash
# Local bootstrap evaluation (mirrors official procedure)
python scripts/evaluate.py \
    paths.checkpoint_dir=outputs/.../checkpoints \
    paths.results_dir=outputs/.../results
```

## Docker Submission

**Step 1:** Copy weights and calibration artifacts into `weights/`:
```bash
cp outputs/YYYY-MM-DD/checkpoints/epoch_*.pt weights/
mkdir -p weights/calibration
cp outputs/YYYY-MM-DD/results/isotonic_calibrator.pkl weights/calibration/
cp outputs/YYYY-MM-DD/results/calibration_results.json weights/calibration/
```

**Step 2:** Update threshold in `docker/Dockerfile`:
```dockerfile
ENV RARE26_THRESHOLD=0.42   # ← replace with your calibrated value
```

**Step 3:** Build and test:
```bash
./do_test_run.sh              # Test with internet
./do_test_run.sh --network-none  # Test without internet (mirrors GC env)
```

**Step 4:** Export and upload:
```bash
./do_save.sh rare26_submission.tar.gz
# Upload rare26_submission.tar.gz to Grand Challenge → Submit
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Isotonic Regression over Temperature Scaling | Non-parametric, no distributional assumption needed with 158 positives |
| Strict val_selection / val_calibration split | Prevents information leakage: model selection ≠ threshold optimization |
| Lightweight head (768→256→1) | Avoids overfitting on 158 positives (previous 768→512→256 was over-parameterized) |
| CV over gamma_neg ∈ {2,4,6} | 158 positives → high variance, don't fix hyperparameters without validation |
| Bootstrap threshold search (1000 iters) | Directly mirrors official evaluation procedure |
| TTA 8 views (deterministic) | No randomness at inference time — reproducible Docker output |

## Submission Rules Checklist

- [ ] Code published under **MIT license**
- [ ] PDF technical paper (2–3 pages) prepared
- [ ] Docker container tested with `--network=none`
- [ ] Single submission to Closed Testing Phase
- [ ] Email sent to `rare-challenge@tue.nl` with team info and algorithm link
- [ ] Account verified on Grand Challenge (do this early!)
