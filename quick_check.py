"""
Run this once on a real CUDA machine to settle the one open question from
app/engines/da3/engine.py's module docstring: does `prediction.depth` come
back at `process_res` or already resampled to the source image's resolution?

    python quick_check.py

Uses a deliberately non-square, non-504 test image so the two possible
outcomes are easy to tell apart in the printed shapes.
"""

from __future__ import annotations

import numpy as np

from app.engines.da3.engine import DA3Engine


def main() -> None:
    engine = DA3Engine()
    print(f"Loading {engine.checkpoint} ...")
    engine.load()
    print(f"Loaded on device: {engine._device}")

    test_image = np.random.randint(0, 256, size=(900, 600, 3), dtype=np.uint8)
    prediction = engine.predict(test_image)

    print(f"source_resolution : {prediction.source_resolution}")
    print(f"raw_depth.shape   : {prediction.raw_depth.shape}")
    print(f"inference_seconds : {prediction.inference_seconds:.3f}")

    if prediction.raw_depth.shape[:2] == prediction.source_resolution:
        print(
            "\n-> raw_depth matches source resolution. Either DA3 already "
            "reconstructs to source size internally, or engine.py's resize "
            "fallback fired (it would either way -- add a print in "
            "DA3Engine.predict() if you need to tell which)."
        )
    else:
        print("\n-> Unexpected: shapes still don't match after engine.py's resize step. Investigate.")


if __name__ == "__main__":
    main()
