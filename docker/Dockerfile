FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="rare26-team"
LABEL description="RARE26 Challenge — Barrett Neoplasia Detection"
LABEL challenge="RARE26"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY inference.py .


COPY weights/ weights/

RUN mkdir -p /input /output
ENV INPUT_DIR=/input
ENV OUTPUT_DIR=/output
ENV WEIGHTS_DIR=/opt/ml/weights

ENV RARE26_THRESHOLD=0.42

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4

ENTRYPOINT ["python", "inference.py"]
