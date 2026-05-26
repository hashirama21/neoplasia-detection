# Weights Directory

Place pretrained and fine-tuned model weights here before building the Docker image.

## Required files

```
weights/
├── dinov2_gastronet5m.pth          # GastroNet-5M pretrained ViT-Base
│                                   # Download: https://cortex.thetavision.nl/dataset-provider/listing/2/
│                                   # HuggingFace: BONS-AI-TUE-AMC/GastroNetDinov2
│
├── model_1.pt                      # Fine-tuned ensemble model 1 (from training)
├── model_2.pt                      # Fine-tuned ensemble model 2
├── model_3.pt                      # Fine-tuned ensemble model 3
├── model_4.pt                      # Fine-tuned ensemble model 4
├── model_5.pt                      # Fine-tuned ensemble model 5
│
└── calibration/
    ├── isotonic_calibrator.pkl     # Fitted IsotonicRegression (from calibration step)
    └── calibration_results.json   # Optimal threshold + bootstrap metrics
```

## How to populate

1. Download GastroNet-5M checkpoint from consortium
2. Run training: `python scripts/train.py`
3. Copy top-k checkpoints: `cp outputs/.../checkpoints/epoch_*.pt weights/`
4. Run calibration: `python scripts/calibrate.py`
5. Copy calibration artifacts: `cp outputs/.../results/* weights/calibration/`
6. Update RARE26_THRESHOLD in docker/Dockerfile with value from calibration_results.json

## .gitignore

All .pt and .pkl files are gitignored — do NOT commit model weights to git.
