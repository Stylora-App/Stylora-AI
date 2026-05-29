import argparse
import base64
import contextlib
import io
import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SWAGGER_UI_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>CLIP Embedding Worker</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({url:"/openapi.yaml",dom_id:"#swagger-ui",
presets:[SwaggerUIBundle.presets.apis,SwaggerUIBundle.SwaggerUIStandalonePreset],
layout:"BaseLayout"})
</script></body></html>"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.set_verbosity_error()


@contextlib.contextmanager
def redirect_native_output_to_devnull():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(sys.stdout.fileno())
    saved_stderr_fd = os.dup(sys.stderr.fileno())
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, sys.stdout.fileno())
        os.dup2(devnull_fd, sys.stderr.fileno())
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, sys.stdout.fileno())
        os.dup2(saved_stderr_fd, sys.stderr.fileno())
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def load_worker(model_id: str):
    with redirect_native_output_to_devnull():
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)

    model.eval()
    return model, processor


def embed_image(model, processor, payload: dict):
    image_base64 = payload.get("imageBase64")
    if not image_base64:
        raise ValueError("Image data is required.")

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with redirect_native_output_to_devnull():
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            image_features = model.visual_projection(pooled_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features[0].cpu().tolist()


class EmbeddingRequestHandler(BaseHTTPRequestHandler):
    server_version = "ClipEmbeddingWorker/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ready": True, "dimensions": self.server.projection_dim})
            return

        if self.server.dev:
            if self.path == "/openapi.yaml":
                self._send_spec()
                return
            if self.path == "/docs":
                self._send_bytes(200, SWAGGER_UI_HTML, "text/html; charset=utf-8")
                return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if self.path != "/embed":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            payload = self._read_json_body()
            embedding = embed_image(self.server.model, self.server.processor, payload)
            self._send_json(200, {"embedding": embedding})
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": str(exc)})

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(body.decode("utf-8"))

    def _send_json(self, status_code: int, payload: dict):
        self._send_bytes(status_code, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_bytes(self, status_code: int, body: bytes, content_type: str):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_spec(self):
        spec_file = self.server.spec_file
        if not spec_file.exists():
            self._send_json(404, {"error": f"Spec file not found: {spec_file}"})
            return
        self._send_bytes(200, spec_file.read_bytes(), "text/yaml; charset=utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--dev", action="store_true", help="Enable /docs and /openapi.yaml routes")
    args = parser.parse_args()

    model, processor = load_worker(args.model_id)
    projection_dim = getattr(model.config, "projection_dim", 512)

    spec_file = pathlib.Path(__file__).parent.parent / "docs" / "clip-openapi.yaml"

    server = ThreadingHTTPServer(("0.0.0.0", args.port), EmbeddingRequestHandler)
    server.model = model
    server.processor = processor
    server.projection_dim = projection_dim
    server.dev = args.dev
    server.spec_file = spec_file

    if args.dev:
        print(f"Dev mode: Swagger UI at http://0.0.0.0:{args.port}/docs")

    server.serve_forever()


if __name__ == "__main__":
    main()
