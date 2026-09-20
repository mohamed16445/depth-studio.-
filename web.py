from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.core.depth_data import DepthResult
from app.core.image_io import SUPPORTED_EXTENSIONS, load_source_image
from app.engines.da3.engine import DA3Engine
from app.export.depth_png16 import export_depth_16bit_png

BASE_DIR = Path(__file__).resolve().parent
INDEX = BASE_DIR / "index.html"

app = FastAPI(title="Depth Studio", version="1.0.0")

_engine: DA3Engine | None = None


def get_engine() -> DA3Engine:
    global _engine
    if _engine is None:
        checkpoint = os.getenv("DA3_CHECKPOINT", "depth-anything/da3mono-large")
        _engine = DA3Engine(checkpoint=checkpoint)
    return _engine


@app.get("/healthz")
def healthz():
    return {"ok": True, "engine": "Depth Anything 3"}


@app.get("/")
def home():
    return FileResponse(INDEX, media_type="text/html")


@app.post("/api/generate")
async def generate_depth(file: UploadFile = File(...), direction: str = "normal"):
    if direction not in {"normal", "inverted"}:
        raise HTTPException(status_code=400, detail="direction must be normal or inverted")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported image format")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is larger than 20 MB")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / f"source{suffix}"
            source_path.write_bytes(raw)
            source = load_source_image(source_path)

            prediction = get_engine().predict(source.array_rgb)
            result = DepthResult(prediction=prediction, direction=direction)  # type: ignore[arg-type]
            normalized = result.normalized()

            output_path = Path(tmp) / "depth.png"
            export_depth_16bit_png(normalized, output_path)
            png_bytes = output_path.read_bytes()

            headers = {
                "X-Depth-Width": str(source.width),
                "X-Depth-Height": str(source.height),
                "X-Inference-Seconds": f"{prediction.inference_seconds:.3f}",
                "Content-Disposition": 'attachment; filename="depth-map-16bit.png"',
            }
            return Response(content=png_bytes, media_type="image/png", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Depth generation failed: {exc}") from exc


# Keep the project usable as a simple static website when opened from a static host.
# The real generator endpoint is provided by this FastAPI server.
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")
