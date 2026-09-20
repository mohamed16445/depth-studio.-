# Depth Studio Web

The original Depth Studio visual interface is preserved. The existing Download CTA now opens the mobile depth generator.

## Architecture

Phone browser → Depth Studio website → FastAPI backend → Depth Anything 3 → 16-bit PNG → phone download.

The DA3 model runs on the server. It is **not** downloaded to or executed on the user's phone.

## Run

This project expects the official Depth Anything 3 source package to be installed because `DA3Engine` imports `depth_anything_3`.

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e .
cd ../depth-studio
pip install -r requirements.txt
./start_web.sh
```

Open the server URL in a phone browser. For real DA3 inference, deploy the backend on hardware supported by the DA3 runtime. Do not commit model checkpoint files to this repository.
