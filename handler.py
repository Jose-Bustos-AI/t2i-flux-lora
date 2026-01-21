import runpod
import os
import websocket
import json
import uuid
import logging
import urllib.request
import subprocess
import time
import random
from urllib.error import HTTPError

import requests  # subir a Supabase + descargar LoRA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

COMFY_ROOT = os.getenv("COMFY_ROOT", "/comfyui")
LORAS_DIR = os.path.join(COMFY_ROOT, "models", "loras")

HANDLER_VERSION = os.getenv("HANDLER_VERSION", "t2i-flux-lora-upload-to-supabase-v1")
print(f"HANDLER VERSION: {HANDLER_VERSION}", flush=True)

DEFAULT_CHECKPOINT = os.getenv("FLUX_CHECKPOINT", "flux1-dev-fp8.safetensors")  # cámbialo en env si tu nombre difiere
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 24
DEFAULT_CFG = 1.2
DEFAULT_GUIDANCE = 3.5

# -------------------------
# Supabase upload (bytes -> public url)
# -------------------------
def supabase_upload_bytes(content: bytes, filename: str, content_type: str = "image/png") -> str:
    """
    Upload bytes to Supabase Storage using REST.
    Requires env vars in RunPod endpoint:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY
    Optional:
      SUPABASE_BUCKET (default results)
      SUPABASE_PATH_PREFIX (default runpod/lora-image)
    """
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    bucket = os.environ.get("SUPABASE_BUCKET", "results")
    prefix = os.environ.get("SUPABASE_PATH_PREFIX", "runpod/lora-image").strip("/")

    path = f"{prefix}/{time.strftime('%Y/%m/%d')}/{filename}"
    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{path}"

    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    r = requests.post(upload_url, headers=headers, params={"upsert": "true"}, data=content, timeout=300)
    if r.status_code not in (200, 201):
        raise Exception(f"Supabase upload failed: {r.status_code} {r.text}")

    public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{path}"
    return public_url


# -------------------------
# Helpers
# -------------------------
def to_nearest_multiple_of_16(value):
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height no es numérico: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def _pick_seed(seed_val):
    try:
        if seed_val is None:
            return random.randint(0, 2**31 - 1)
        seed_int = int(seed_val)
        if seed_int < 0:
            return random.randint(0, 2**31 - 1)
        return seed_int
    except Exception:
        return random.randint(0, 2**31 - 1)


def _ensure_comfy_ready():
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    max_http_attempts = 180
    for attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP OK (attempt {attempt+1})")
            return
        except Exception as e:
            logger.warning(f"HTTP not ready ({attempt+1}/{max_http_attempts}): {e}")
            time.sleep(1)
    raise Exception("No puedo conectar con ComfyUI en 8188. ¿Está arrancado?")


def _connect_ws():
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting WS: {ws_url}")
    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WS OK (attempt {attempt+1})")
            return ws
        except Exception as e:
            logger.warning(f"WS not ready ({attempt+1}/{max_attempts}): {e}")
            time.sleep(5)
    raise Exception("WS timeout (3 min)")


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "(could not read body)"
        raise Exception(f"ComfyUI /prompt returned {e.code}: {body}")


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def wait_until_done(ws, prompt_id):
    """
    Wait until ComfyUI finishes prompt_id (executing node=None).
    """
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return
        # ignore binary frames


def comfy_view_download(filename: str, subfolder: str = "", filetype: str = "output") -> bytes:
    """
    Download a file from ComfyUI /view endpoint.
    In history outputs, images entries look like:
      {"filename":"xxx.png","subfolder":"","type":"output"}
    """
    qs = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder or "",
        "type": filetype or "output",
    })
    url = f"http://{server_address}:8188/view?{qs}"
    logger.info(f"Downloading from Comfy /view: {url}")
    with urllib.request.urlopen(url) as resp:
        return resp.read()


# -------------------------
# LoRA download (URL -> /comfyui/models/loras)
# -------------------------
def ensure_lora(lora_url: str, lora_name: str = None) -> str:
    """
    Download LoRA to LORAS_DIR. Returns local filename (basename).
    """
    if not lora_url:
        return None

    os.makedirs(LORAS_DIR, exist_ok=True)

    if not lora_name:
        # fallback name from URL
        base = os.path.basename(lora_url.split("?", 1)[0])
        lora_name = base if base else f"lora_{uuid.uuid4()}.safetensors"

    if not lora_name.endswith(".safetensors"):
        # allow user to pass without extension
        lora_name = lora_name + ".safetensors"

    dest_path = os.path.join(LORAS_DIR, lora_name)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        logger.info(f"LoRA already exists: {dest_path}")
        return lora_name

    logger.info(f"Downloading LoRA: {lora_url} -> {dest_path}")
    with requests.get(lora_url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise Exception(f"LoRA download failed: {r.status_code} {r.text[:500]}")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if os.path.getsize(dest_path) < 1024:
        raise Exception("LoRA descargada demasiado pequeña (posible URL inválida).")

    return lora_name


# -------------------------
# Minimal FLUX workflow builder (fallback)
# -------------------------
def build_flux_workflow(prompt_text: str,
                        negative_text: str,
                        width: int,
                        height: int,
                        steps: int,
                        cfg: float,
                        guidance: float,
                        seed: int,
                        checkpoint_name: str,
                        lora_file: str = None,
                        lora_strength_model: float = 1.0,
                        lora_strength_clip: float = 1.0) -> dict:
    """
    Minimal workflow for FLUX-like setups.
    IMPORTANT: Requires that your ComfyUI image has the proper FLUX nodes available.
    If your environment doesn't have FluxGuidance etc., you should pass input.workflow instead.
    """
    # Node IDs as strings (Comfy format)
    N = lambda i: str(i)

    wf = {}

    # 10: Load checkpoint (model+clip+vae)
    wf[N(10)] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": checkpoint_name
        }
    }

    # 20: Positive prompt
    wf[N(20)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [N(10), 1],
            "text": prompt_text
        }
    }

    # 21: Negative prompt
    wf[N(21)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [N(10), 1],
            "text": negative_text
        }
    }

    # 30: Latent
    # For many modern models, this node exists; if not, you must use your own workflow.
    wf[N(30)] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1
        }
    }

    model_ref = [N(10), 0]
    clip_ref = [N(10), 1]

    # 40: Optional LoRA
    if lora_file:
        wf[N(40)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_file,
                "strength_model": float(lora_strength_model),
                "strength_clip": float(lora_strength_clip),
            }
        }
        model_ref = [N(40), 0]
        clip_ref = [N(40), 1]

        # Re-encode prompts with LoRA clip (optional but safer)
        wf[N(20)]["inputs"]["clip"] = clip_ref
        wf[N(21)]["inputs"]["clip"] = clip_ref

    # 50: Flux guidance (if exists in your build)
    wf[N(50)] = {
        "class_type": "FluxGuidance",
        "inputs": {
            "conditioning": [N(20), 0],
            "guidance": float(guidance)
        }
    }

    cond_pos_ref = [N(50), 0]
    cond_neg_ref = [N(21), 0]

    # 60: Sampler
    wf[N(60)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": cond_pos_ref,
            "negative": cond_neg_ref,
            "latent_image": [N(30), 0],
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    }

    # 70: Decode VAE
    wf[N(70)] = {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [N(60), 0],
            "vae": [N(10), 2]
        }
    }

    # 80: Save image
    wf[N(80)] = {
        "class_type": "SaveImage",
        "inputs": {
            "images": [N(70), 0],
            "filename_prefix": "runpod_t2i"
        }
    }

    return wf


# -------------------------
# Extract first image from history outputs
# -------------------------
def extract_first_image_ref(history_for_prompt: dict):
    """
    Returns (filename, subfolder, type) from Comfy history, or None.
    """
    outputs = history_for_prompt.get("outputs", {}) or {}
    for node_id, node_out in outputs.items():
        if isinstance(node_out, dict) and "images" in node_out and isinstance(node_out["images"], list):
            for img in node_out["images"]:
                fn = img.get("filename")
                if fn:
                    return fn, img.get("subfolder", ""), img.get("type", "output")
    return None


def handler(job):
    job_input = job.get("input", {}) or {}
    logger.info(f"Received job input keys: {list(job_input.keys())} | version={HANDLER_VERSION}")

    # Required: prompt
    user_prompt = job_input.get("prompt")
    if not user_prompt or not isinstance(user_prompt, str):
        raise Exception("Missing required field: prompt (string)")

    negative_default = "blurry, lowres, deformed, extra fingers, bad hands, bad face, jpeg artifacts"
    negative_prompt = job_input.get("negative_prompt", negative_default)

    # params
    seed = _pick_seed(job_input.get("seed"))
    width = to_nearest_multiple_of_16(job_input.get("width", DEFAULT_WIDTH))
    height = to_nearest_multiple_of_16(job_input.get("height", DEFAULT_HEIGHT))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    cfg = float(job_input.get("cfg", DEFAULT_CFG))
    guidance = float(job_input.get("guidance", DEFAULT_GUIDANCE))
    checkpoint = job_input.get("checkpoint", DEFAULT_CHECKPOINT)

    # LoRA optional
    lora_url = job_input.get("lora_url") or ""
    lora_name = job_input.get("lora_name") or None
    lora_strength_model = float(job_input.get("lora_strength_model", 1.0))
    lora_strength_clip = float(job_input.get("lora_strength_clip", 1.0))

    lora_file = None
    if lora_url:
        lora_file = ensure_lora(lora_url, lora_name)

    # Workflow: if user provides, use it. Else build minimal.
    workflow = job_input.get("workflow")
    if workflow and isinstance(workflow, dict):
        prompt_wf = workflow
        logger.info("Using input.workflow provided by caller.")
    else:
        prompt_wf = build_flux_workflow(
            prompt_text=user_prompt,
            negative_text=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            guidance=guidance,
            seed=seed,
            checkpoint_name=checkpoint,
            lora_file=lora_file,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip
        )
        logger.info("Using built-in minimal FLUX workflow (fallback).")

    # Run Comfy
    _ensure_comfy_ready()
    ws = _connect_ws()
    try:
        resp = queue_prompt(prompt_wf)
        prompt_id = resp["prompt_id"]
        wait_until_done(ws, prompt_id)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    # Get history + download first image
    hist = get_history(prompt_id)
    if prompt_id not in hist:
        raise Exception("Comfy history missing prompt_id")

    ref = extract_first_image_ref(hist[prompt_id])
    if not ref:
        raise Exception("No image found in Comfy outputs (history).")

    filename, subfolder, ftype = ref
    img_bytes = comfy_view_download(filename, subfolder, ftype)

    # Upload to Supabase
    out_name = f"{uuid.uuid4()}.png"
    image_url = supabase_upload_bytes(img_bytes, out_name, "image/png")

    logger.info(f"RETURNING image_url: {image_url} | version={HANDLER_VERSION}")

    return {
        "image_url": image_url,
        "seed_used": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "guidance": guidance,
        "checkpoint": checkpoint,
        "lora_name": lora_file or "",
        "version": HANDLER_VERSION
    }


runpod.serverless.start({"handler": handler})
