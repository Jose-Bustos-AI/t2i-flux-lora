import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import time
import random
from urllib.error import HTTPError

import requests  # <-- necesario para subir a Supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

COMFY_ROOT = os.getenv("COMFY_ROOT", "/ComfyUI")
COMFY_INPUT_DIR = os.path.join(COMFY_ROOT, "input")

# Optional but HIGHLY recommended stamp (no rompe nada)
HANDLER_VERSION = os.getenv("HANDLER_VERSION", "img2vid-upload-to-supabase-v1")
print(f"HANDLER VERSION: {HANDLER_VERSION}", flush=True)

# -------------------------
# Supabase upload (bytes -> public url)
# -------------------------
def supabase_upload_bytes(content: bytes, filename: str, content_type: str = "video/mp4") -> str:
    """
    Upload bytes to Supabase Storage using REST.
    Requires env vars in RunPod endpoint:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY
    Optional:
      SUPABASE_BUCKET (default results)
      SUPABASE_PATH_PREFIX (default runpod)
    """
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    bucket = os.environ.get("SUPABASE_BUCKET", "results")
    prefix = os.environ.get("SUPABASE_PATH_PREFIX", "runpod").strip("/")

    path = f"{prefix}/{time.strftime('%Y/%m/%d')}/{filename}"
    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{path}"

    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    # Supabase storage object upload: use POST with upsert
    r = requests.post(upload_url, headers=headers, params={"upsert": "true"}, data=content, timeout=300)
    if r.status_code not in (200, 201):
        raise Exception(f"Supabase upload failed: {r.status_code} {r.text}")

    public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{path}"
    return public_url


def to_nearest_multiple_of_16(value):
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height 값이 숫자가 아닙니다: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def download_file_from_url(url, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"✅ Downloaded: {url} -> {output_path}")
            return output_path
        raise Exception(f"wget failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        raise Exception(f"다운로드 중 오류 발생: {e}")


def save_base64_to_file(base64_data, output_path):
    """
    Supports:
      - raw base64
      - data-uri: data:...;base64,....
      - auto padding
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not isinstance(base64_data, str):
            raise Exception("base64_data is not a string")

        b64 = base64_data.strip()
        if "base64," in b64:
            b64 = b64.split("base64,", 1)[1].strip()

        missing = (-len(b64)) % 4
        if missing:
            b64 += "=" * missing

        decoded_data = base64.b64decode(b64, validate=False)
        with open(output_path, 'wb') as f:
            f.write(decoded_data)

        logger.info(f"✅ Saved base64 to: {output_path}")
        return output_path
    except (binascii.Error, ValueError) as e:
        raise Exception(f"Base64 디코딩 실패: {e}")


def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
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
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_video_paths(ws, prompt):
    """
    Instead of returning base64, return file paths from ComfyUI history outputs.
    We will upload the first mp4 found to Supabase.
    """
    prompt_id = queue_prompt(prompt)['prompt_id']

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message.get('type') == 'executing':
                data = message.get('data', {})
                if data.get('node') is None and data.get('prompt_id') == prompt_id:
                    break
        # ignore binary frames

    history = get_history(prompt_id)[prompt_id]
    outputs = history.get('outputs', {})

    video_paths_by_node = {}

    for node_id, node_output in outputs.items():
        paths = []

        # Many workflows put mp4/gif under "gifs"
        if isinstance(node_output, dict) and "gifs" in node_output and isinstance(node_output["gifs"], list):
            for item in node_output["gifs"]:
                fullpath = item.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    paths.append(fullpath)

        # Some workflows may use "videos"
        if isinstance(node_output, dict) and "videos" in node_output and isinstance(node_output["videos"], list):
            for item in node_output["videos"]:
                fullpath = item.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    paths.append(fullpath)

        # Some workflows may use "images" for sequences, but here we want mp4 only.
        video_paths_by_node[str(node_id)] = paths

    return video_paths_by_node


def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def _ensure_comfy_ready():
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    max_http_attempts = 180
    for attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {attempt+1})")
            return
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {attempt+1}/{max_http_attempts}): {e}")
            time.sleep(1)
    raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")


def _connect_ws():
    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            return ws
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            time.sleep(5)
    raise Exception("웹소켓 연결 시간 초과 (3분)")


def _pick_seed(seed_val):
    # WanVideoSampler in your workflow rejects seed < 0.
    try:
        if seed_val is None:
            return random.randint(0, 2**31 - 1)
        seed_int = int(seed_val)
        if seed_int < 0:
            return random.randint(0, 2**31 - 1)
        return seed_int
    except Exception:
        return random.randint(0, 2**31 - 1)


def _materialize_input_image(job_input, task_id):
    """
    Preferred: RunPod-style job_input.images[0] = {name,url}
    Fallback: image_url/image_base64/image_path
    Returns (filename_in_comfy_input, full_path)
    """
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

    # 1) RunPod images list
    images = job_input.get("images")
    if isinstance(images, list) and len(images) > 0 and isinstance(images[0], dict):
        name = images[0].get("name") or "input_0.png"
        url = images[0].get("url")
        if not url:
            raise Exception("images[0].url is missing")
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        download_file_from_url(url, full_path)
        return name, full_path

    # 2) Legacy keys
    if "image_url" in job_input:
        name = "input_0.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        download_file_from_url(job_input["image_url"], full_path)
        return name, full_path

    if "image_base64" in job_input:
        name = "input_0.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        save_base64_to_file(job_input["image_base64"], full_path)
        return name, full_path

    if "image_path" in job_input:
        name = "input_0.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        subprocess.run(["cp", "-f", job_input["image_path"], full_path], check=True)
        return name, full_path

    # 3) Default demo
    name = "example_image.png"
    full_path = os.path.join(COMFY_INPUT_DIR, name)
    if os.path.exists("/example_image.png"):
        subprocess.run(["cp", "-f", "/example_image.png", full_path], check=True)
        logger.info("기본 이미지 파일을 사용합니다: /example_image.png")
        return name, full_path

    raise Exception("No image provided. Send job_input.images[0].url (preferred) or image_url/image_base64/image_path.")


def _materialize_end_image_if_any(job_input):
    """
    Optional end image (FLF2V)
    Returns (filename, full_path) or (None, None)
    """
    os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

    if "end_image_url" in job_input:
        name = "end_image.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        download_file_from_url(job_input["end_image_url"], full_path)
        return name, full_path

    if "end_image_base64" in job_input:
        name = "end_image.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        save_base64_to_file(job_input["end_image_base64"], full_path)
        return name, full_path

    if "end_image_path" in job_input:
        name = "end_image.png"
        full_path = os.path.join(COMFY_INPUT_DIR, name)
        subprocess.run(["cp", "-f", job_input["end_image_path"], full_path], check=True)
        return name, full_path

    return None, None


def handler(job):
    job_input = job.get("input", {}) or {}
    logger.info(f"Received job input keys: {list(job_input.keys())} | version={HANDLER_VERSION}")

    # Required prompt
    user_prompt = job_input.get("prompt")
    if not user_prompt or not isinstance(user_prompt, str):
        raise Exception("Missing required field: prompt (string)")

    task_id = f"task_{uuid.uuid4()}"

    # Image in
    image_name, _image_full = _materialize_input_image(job_input, task_id)

    # Optional end image (FLF2V)
    end_image_name, _end_full = _materialize_end_image_if_any(job_input)

    # Workflow file
    workflow_file = "/new_Wan22_flf2v_api.json" if end_image_name else "/new_Wan22_api.json"
    logger.info(f"Using {'FLF2V' if end_image_name else 'single'} workflow: {workflow_file}")

    prompt = load_workflow(workflow_file)

    # Defaults + sanitize
    length = int(job_input.get("length", 81))
    steps = int(job_input.get("steps", 10))
    cfg = float(job_input.get("cfg", 7))
    seed = _pick_seed(job_input.get("seed"))

    width_in = job_input.get("width", 832)
    height_in = job_input.get("height", 480)
    adjusted_width = to_nearest_multiple_of_16(width_in)
    adjusted_height = to_nearest_multiple_of_16(height_in)

    negative_default = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    negative_prompt = job_input.get("negative_prompt", negative_default)

    # ---- Apply to your known node IDs ----
    prompt["244"]["inputs"]["image"] = image_name
    prompt["541"]["inputs"]["num_frames"] = length
    prompt["135"]["inputs"]["positive_prompt"] = user_prompt
    prompt["135"]["inputs"]["negative_prompt"] = negative_prompt

    # seed + cfg
    prompt["220"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["cfg"] = cfg

    # width/height
    prompt["235"]["inputs"]["value"] = adjusted_width
    prompt["236"]["inputs"]["value"] = adjusted_height

    # context
    prompt["498"]["inputs"]["context_overlap"] = int(job_input.get("context_overlap", 48))
    prompt["498"]["inputs"]["context_frames"] = length

    # steps section (if exists)
    if "834" in prompt:
        prompt["834"]["inputs"]["steps"] = steps
        lowsteps = int(steps * 0.6)
        if "829" in prompt:
            prompt["829"]["inputs"]["step"] = lowsteps
        logger.info(f"Steps set to: {steps} | LowSteps: {lowsteps}")

    # end image (FLF2V)
    if end_image_name:
        prompt["617"]["inputs"]["image"] = end_image_name

    # Ensure Comfy is up, run
    _ensure_comfy_ready()
    ws = _connect_ws()
    try:
        video_paths_by_node = get_video_paths(ws, prompt)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    # Find first existing video path and upload
    for node_id, paths in video_paths_by_node.items():
        for p in paths:
            if p and os.path.exists(p):
                logger.info(f"Found video file: {p} (node {node_id})")
                with open(p, "rb") as f:
                    mp4_bytes = f.read()

                filename = f"{uuid.uuid4()}.mp4"
                video_url = supabase_upload_bytes(mp4_bytes, filename, "video/mp4")

                logger.info(f"RETURNING video_url: {video_url} | version={HANDLER_VERSION}")
                return {
                    "video_url": video_url,
                    "seed_used": seed,
                    "node_id": node_id,
                    "width": adjusted_width,
                    "height": adjusted_height,
                    "length": length
                }

    return {
        "error": "비디오를 찾을 수 없습니다. (No video file in outputs)",
        "seed_used": seed,
        "version": HANDLER_VERSION
    }


runpod.serverless.start({"handler": handler})



