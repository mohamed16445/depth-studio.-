from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from depth_anything_3.api import DepthAnything3


# ============================================================
# Depth Studio
# Depth Anything 3 backend
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX = BASE_DIR / "index.html"

app = FastAPI(
    title="Depth Studio",
    version="2.0.0",
)

# ------------------------------------------------------------
# Model settings
# ------------------------------------------------------------

MODEL_NAME = os.getenv(
    "DA3_MODEL",
    "depth-anything/DA3-LARGE",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: DepthAnything3 | None = None


# ------------------------------------------------------------
# Load Depth Anything 3
# ------------------------------------------------------------

def get_model() -> DepthAnything3:
    global _model

    if _model is None:
        print(f"[Depth Studio] Loading Depth Anything 3...")
        print(f"[Depth Studio] Model: {MODEL_NAME}")
        print(f"[Depth Studio] Device: {DEVICE}")

        _model = DepthAnything3.from_pretrained(MODEL_NAME)
        _model = _model.to(DEVICE)
        _model.eval()

        print("[Depth Studio] Depth Anything 3 loaded successfully.")

    return _model


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "engine": "Depth Anything 3",
        "model": MODEL_NAME,
        "device": DEVICE,
    }


# ------------------------------------------------------------
# Homepage
# ------------------------------------------------------------

@app.get("/")
def home():
    if not INDEX.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found",
        )

    return FileResponse(
        INDEX,
        media_type="text/html",
    )


# ------------------------------------------------------------
# Normalize depth
# ------------------------------------------------------------

def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Convert Depth Anything 3 output into
    a clean 16-bit grayscale depth map.
    """

    depth = np.asarray(depth, dtype=np.float32)

    # Remove invalid values
    finite = np.isfinite(depth)

    if not np.any(finite):
        raise ValueError("Depth prediction contains no valid values.")

    valid = depth[finite]

    minimum = float(valid.min())
    maximum = float(valid.max())

    # Avoid division by zero
    if maximum - minimum < 1e-8:
        return np.zeros(depth.shape, dtype=np.uint16)

    depth = np.nan_to_num(
        depth,
        nan=minimum,
        posinf=maximum,
        neginf=minimum,
    )

    depth = (depth - minimum) / (maximum - minimum)

    depth = np.clip(
        depth,
        0.0,
        1.0,
    )

    return (depth * 65535.0).astype(np.uint16)


# ------------------------------------------------------------
# Generate depth map
# ------------------------------------------------------------

@app.post("/api/generate")
async def generate_depth(
    file: UploadFile = File(...),
    direction: str = "normal",
):
    if direction not in {"normal", "inverted"}:
        raise HTTPException(
            status_code=400,
            detail="direction must be normal or inverted",
        )

    filename = file.filename or ""

    suffix = Path(filename).suffix.lower()

    supported = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    if suffix not in supported:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format",
        )

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    # 20 MB maximum upload
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is larger than 20 MB",
        )

    start_time = time.perf_counter()

    try:
        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

        original_width, original_height = image.size

        # ----------------------------------------------------
        # Get model
        # ----------------------------------------------------

        model = get_model()

        # ----------------------------------------------------
        # Run Depth Anything 3
        # ----------------------------------------------------

        with torch.inference_mode():

            prediction = model.inference(
                images=[image],
                process_res=504,
            )

        # ----------------------------------------------------
        # Get depth
        # ----------------------------------------------------

        depth = prediction.depth[0]

        depth = np.asarray(
            depth,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Resize depth back to original image size
        # ----------------------------------------------------

        depth_image = Image.fromarray(
            depth,
            mode="F",
        )

        depth_image = depth_image.resize(
            (
                original_width,
                original_height,
            ),
            Image.Resampling.BILINEAR,
        )

        depth = np.asarray(
            depth_image,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Normalize to 16-bit
        # ----------------------------------------------------

        depth16 = normalize_depth(depth)

        # ----------------------------------------------------
        # Invert if requested
        # ----------------------------------------------------

        if direction == "inverted":
            depth16 = 65535 - depth16

        # ----------------------------------------------------
        # Export PNG
        # ----------------------------------------------------

        output = io.BytesIO()

        Image.fromarray(
            depth16,
            mode="I;16",
        ).save(
            output,
            format="PNG",
        )

        png_bytes = output.getvalue()

        elapsed = time.perf_counter() - start_time

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        headers = {
            "X-Depth-Width": str(original_width),
            "X-Depth-Height": str(original_height),
            "X-Inference-Seconds": f"{elapsed:.3f}",
            "X-Depth-Engine": "Depth Anything 3",
            "X-Depth-Model": MODEL_NAME,
            "Content-Disposition": (
                'attachment; filename="depth-map-16bit.png"'
            ),
        }

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers=headers,
        )

    except HTTPException:
        raise

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=(
                "GPU out of memory. "
                "Try a smaller input image."
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Depth generation failed: {exc}",
        ) from exc
