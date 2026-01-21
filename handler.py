import runpod
import os
import websocket
import json
import uuid
import logging
import urllib.request
import urllib.parse
import time
import random
from urllib.error import HTTPError

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1")
client_id = str(uuid.uuid4())

COMFY_ROOT = os.getenv("COMFY_ROOT", "/comfyui")
LORAS_DIR = os.path.join(COMFY_ROOT, "models", "loras")

HANDLER_VERSION = os.getenv("HANDLER_VERSION", "t2i-flux1dev-lora-supabase-v1")
print(f"HANDLER VERSION: {HANDLER_VERSION}", flush=True)

DEFAULT_CHECKPOINT = os.getenv("FLUX_CHECKPOINT", "flux1-dev-fp8.safetensors")

# -------------------------
# Supabase upload
# -------------------------
def supabase_upload_bytes(content: bytes, filename: str, content_type: str = "image/png") -> str:
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

    return f"{supabase_url}/storage/v1/object/public/{bucket}/{path}"


# -------------------------
# Helpers
# -------------------------
def to_nearest_multiple_of_16(value):
    try:
        v = float(value)
    except Exception:
        raise Exception(f"width/height no es numérico: {value}")
    adjusted = int(round(v / 16.0) * 16)
    return max(adjusted, 16)


def _pick_seed(seed_val):
    try:
        if seed_val is None:
            return random.randint(0, 2**31 - 1)
        s = int(seed_val)
        return random.randint(0, 2**31 - 1) if s < 0 else s
    except Exception:
        return random.randint(0, 2**31 - 1)


def _ensure_comfy_ready():
    http_url = f"http://{server_address}:8188/"
    for i in range(180):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("ComfyUI HTTP ready")
            return
        except Exception:
            time.sleep(1)
    raise Exception("ComfyUI no responde en 8188")


def _connect_ws():
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    ws = websocket.WebSocket()
    for i in range(36):
        try:
            ws.connect(ws_url)
            logger.info("ComfyUI WS connected")
            return ws
        except Exception:
            time.sleep(5)
    raise Exception("WS timeout")


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    payload = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = "(no body)"
        raise Exception(f"/prompt {e.code}: {body}")


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def wait_until_done(ws, prompt_id):
    while True:
        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            if msg.get("type") == "executing":
                data = msg.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return


def comfy_view_download(filename: str, subfolder: str = "", filetype: str = "output") -> bytes:
    qs = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder or "",
        "type": filetype or "output",
    })
    url = f"http://{server_address}:8188/view?{qs}"
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def extract_first_image_ref(history_for_prompt: dict):
    outputs = history_for_prompt.get("outputs", {}) or {}
    for _, node_out in outputs.items():
        if isinstance(node_out, dict) and "images" in node_out and isinstance(node_out["images"], list):
            for img in node_out["images"]:
                fn = img.get("filename")
                if fn:
                    return fn, img.get("subfolder", ""), img.get("type", "output")
    return None


def ensure_lora(lora_url: str, lora_name: str = None) -> str:
    if not lora_url:
        return None

    os.makedirs(LORAS_DIR, exist_ok=True)

    if not lora_name:
        base = os.path.basename(lora_url.split("?", 1)[0])
        lora_name = base if base else f"lora_{uuid.uuid4()}.safetensors"

    if not lora_name.endswith(".safetensors"):
        lora_name = lora_name + ".safetensors"

    dest_path = os.path.join(LORAS_DIR, lora_name)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        logger.info(f"LoRA ya existe: {dest_path}")
        return lora_name

    logger.info(f"Descargando LoRA: {lora_url} -> {dest_path}")
    with requests.get(lora_url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise Exception(f"LoRA download failed: {r.status_code} {r.text[:300]}")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if os.path.getsize(dest_path) < 1024:
        raise Exception("LoRA descargada demasiado pequeña (URL mala?)")

    return lora_name


def build_flux_workflow(prompt_text, negative_text, width, height, steps, cfg, guidance, seed,
                        checkpoint_name, lora_file=None, lora_strength_model=1.0, lora_strength_clip=1.0):
    """
    Workflow mínimo para FLUX. Requiere que tu ComfyUI tenga nodos FLUX/SD3 disponibles.
    """
    N = lambda i: str(i)
    wf = {}

    wf[N(10)] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint_name}}

    model_ref = [N(10), 0]
    clip_ref = [N(10), 1]

    if lora_file:
        wf[N(15)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref,
                "clip": clip_ref,
                "lora_name": lora_file,
                "strength_model": float(lora_strength_model),
                "strength_clip": float(lora_strength_clip),
            }
        }
        model_ref = [N(15), 0]
        clip_ref = [N(15), 1]

    wf[N(20)] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": prompt_text}}
    wf[N(21)] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": negative_text}}

    wf[N(30)] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}

    wf[N(40)] = {"class_type": "FluxGuidance", "inputs": {"conditioning": [N(20), 0], "guidance": float(guidance)}}

    wf[N(60)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": [N(40), 0],
            "negative": [N(21), 0],
            "latent_image": [N(30), 0],
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }
    }

    wf[N(70)] = {"class_type": "VAEDecode", "inputs": {"samples": [N(60), 0], "vae": [N(10), 2]}}
    wf[N(80)] = {"class_type": "SaveImage", "inputs": {"images": [N(70), 0], "filename_prefix": "runpod_t2i"}}

    return wf


def handler(job):
    job_input = job.get("input", {}) or {}
    logger.info(f"keys={list(job_input.keys())} version={HANDLER_VERSION}")

    prompt_text = job_input.get("prompt")
    if not prompt_text or not isinstance(prompt_text, str):
        raise Exception("Missing required field: prompt (string)")

    negative_text = job_input.get("negative_prompt") or "blurry, lowres, deformed, extra fingers"
    seed = _pick_seed(job_input.get("seed"))

    width = to_nearest_multiple_of_16(job_input.get("width", 1024))
    height = to_nearest_multiple_of_16(job_input.get("height", 1024))

    steps = int(job_input.get("steps", 24))
    cfg = float(job_input.get("cfg", 1.2))
    guidance = float(job_input.get("guidance", 3.5))

    checkpoint = job_input.get("checkpoint") or DEFAULT_CHECKPOINT

    lora_url = job_input.get("lora_url") or ""
    lora_name = job_input.get("lora_name") or None
    lora_strength_model = float(job_input.get("lora_strength_model", 1.0))
    lora_strength_clip = float(job_input.get("lora_strength_clip", 1.0))

    lora_file = ensure_lora(lora_url, lora_name) if lora_url else None

    workflow = job_input.get("workflow")
    if workflow and isinstance(workflow, dict):
        prompt_wf = workflow
        logger.info("Using input.workflow")
    else:
        prompt_wf = build_flux_workflow(
            prompt_text, negative_text, width, height, steps, cfg, guidance, seed,
            checkpoint, lora_file, lora_strength_model, lora_strength_clip
        )
        logger.info("Using built-in FLUX workflow")

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

    hist = get_history(prompt_id)
    if prompt_id not in hist:
        raise Exception("history missing prompt_id")

    ref = extract_first_image_ref(hist[prompt_id])
    if not ref:
        raise Exception("No image in outputs")

    filename, subfolder, ftype = ref
    img_bytes = comfy_view_download(filename, subfolder, ftype)

    out_name = f"{uuid.uuid4()}.png"
    image_url = supabase_upload_bytes(img_bytes, out_name, "image/png")

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

