#!/bin/bash
set -e

echo "Starting ComfyUI in the background..."
cd /ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 &

echo "Waiting for ComfyUI to be ready..."
max_wait=240
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

echo "Starting the handler..."
cd /
exec python3 /handler.py
