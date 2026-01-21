FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SERVER_ADDRESS=127.0.0.1
ENV COMFY_ROOT=/ComfyUI
ENV FLUX_CHECKPOINT=flux1-dev-fp8.safetensors

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip install runpod websocket-client requests==2.32.3

# ComfyUI
WORKDIR /
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /ComfyUI
WORKDIR /ComfyUI
RUN pip install -r requirements.txt

# (opcional pero útil) Manager
WORKDIR /ComfyUI/custom_nodes
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
    && pip install -r ComfyUI-Manager/requirements.txt

# Carpetas modelos
RUN mkdir -p /ComfyUI/models/checkpoints /ComfyUI/models/loras

# FLUX.1-dev FP8 checkpoint (1 archivo)
# Fuente: Comfy-Org repo recomendado por docs/ejemplos :contentReference[oaicite:4]{index=4}
RUN wget -q "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
    -O "/ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors"

# App
WORKDIR /
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
