import base64
import io
import os

import numpy as np
import runpod
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3


MODEL_NAME = "depth-anything/DA3-LARGE-1.1"
MAX_IMAGE_BYTES = 20 * 1024 * 1024

model = None


def get_model():
    global model

    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(MODEL_NAME)
        model = model.to(device=device)
        model.eval()

    return model


def decode_image(value):
    if not value:
        raise ValueError("No image was provided.")

    if isinstance(value, str) and value.startswith("data:"):
        value = value.split(",", 1)[1]

    data = base64.b64decode(value)

    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than 20 MB.")

    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_png(array):
    image = Image.fromarray(array, mode="I;16")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})

    image_data = job_input.get("image_base64") or job_input.get("image")
    direction = job_input.get("direction", "normal")

    image = decode_image(image_data)

    original_width, original_height = image.size

    da3 = get_model()

    prediction = da3.inference(
        [image],
        process_res=504,
        process_res_method="upper_bound_resize",
    )

    depth = np.asarray(prediction.depth[0], dtype=np.float32)

    depth_image = Image.fromarray(depth)
    depth_image = depth_image.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR,
    )

    depth = np.asarray(depth_image, dtype=np.float32)

    minimum = float(depth.min())
    maximum = float(depth.max())

    if maximum > minimum:
        depth = (depth - minimum) / (maximum - minimum)
    else:
        depth = np.zeros_like(depth)

    if direction == "inverted":
        depth = 1.0 - depth

    depth_uint16 = np.clip(depth * 65535.0, 0, 65535).astype(np.uint16)

    return {
        "depth_png_base64": encode_png(depth_uint16),
        "width": original_width,
        "height": original_height,
        "format": "PNG",
        "bit_depth": 16,
        "direction": direction,
    }


runpod.serverless.start({"handler": handler})
