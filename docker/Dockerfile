FROM --platform=linux/amd64 pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="rare26-team"
LABEL description="RARE26 — Barrett Neoplasia Detection (DINOv2 ViT-B + LoRA + Ensemble)"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TORCH_HOME=/opt/ml/.torch_cache

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /opt/ml

COPY --chown=user:user requirements.txt .

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user src/       src/
COPY --chown=user:user configs/   configs/
COPY --chown=user:user weights/   weights/
COPY --chown=user:user inference.py .

RUN python -c "
import timm, torch
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=392, dynamic_img_size=True)
out = model(torch.zeros(1, 3, 392, 392))
print(f'Pre-warm OK — output: {out.shape}')
del model
"

ENTRYPOINT ["python", "inference.py"]