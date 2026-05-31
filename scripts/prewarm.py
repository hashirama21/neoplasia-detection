"""Pre-warm timm ViT-B/14 DINOv2 architecture at Docker build time."""
import os, timm, torch

assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE must be set"

model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=False,
    num_classes=0,
    img_size=392,
    dynamic_img_size=True,
)
with torch.no_grad():
    out = model(torch.zeros(1, 3, 392, 392))

print(f"Pre-warm OK — output: {out.shape}, HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}")
del model
