"""Pre-warm timm model architectures at Docker build time.

Instantiates each backbone that may appear in the ensemble so that timm can
download/cache architecture metadata before the container goes offline.
Run by the Dockerfile — must succeed in HF_HUB_OFFLINE=1 mode.
"""
import os
import timm
import torch

assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be set"

for model_name, kwargs in [
    (
        "vit_base_patch14_dinov2.lvd142m",
        {"img_size": 392, "dynamic_img_size": True},
    ),
    (
        "resnet50",
        {},
    ),
]:
    model = timm.create_model(model_name, pretrained=False, num_classes=0, **kwargs)
    with torch.no_grad():
        h = 224 if "resnet" in model_name else 392
        out = model(torch.zeros(1, 3, h, h))
    print(f"Pre-warm OK — {model_name}  output: {out.shape}")
    del model
