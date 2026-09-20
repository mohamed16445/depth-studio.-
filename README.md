# Depth Studio — Milestone 1

Minimal prototype per spec section 24: **Image → DA3 → Depth → Preview → 16-bit PNG export.**
Marigold, the Model Manager, refinement controls, batch processing, and everything else in
the master spec are out of scope on purpose — they're Milestone 2+.

## What was actually verified, and how

This code was written against the *documented* Depth Anything 3 and Marigold APIs, checked
against primary sources rather than taken only from the compatibility report you provided:

- The real `depth_anything_3/api.py` source (`class DepthAnything3(nn.Module,
  PyTorchModelHubMixin)`), a real `depth_anything_3/services/backend.py` (confirms
  `process_res=504`, `process_res_method="upper_bound_resize"` as actual defaults), and the
  official Hugging Face model cards for da3-base / da3-small / da3mono-large /
  da3metric-large / da3-giant — all show an identical `inference()` call shape and identical
  `prediction.depth` / `.conf` / `.extrinsics` / `.intrinsics` return shapes.
- The Marigold `diffusers` core pipeline source (`pipeline_marigold_depth.py`) and the
  `prs-eth/marigold-depth-v1-1` model card directly.

**What this container cannot do:** it has no GPU and no network access to pip-install
torch/xformers/diffusers or download model checkpoints. DA3Engine.load()/predict() (the
actual model call) and the PySide6 UI are therefore untested here — treat those as
carefully-written-but-unrun code, and check them yourself on a real machine before trusting
them.

**What actually was runtime-tested here**, not just syntax-checked — see `tests/test_pipeline.py`,
7/7 passing: image loading is pixel-identical to the source file; depth normalization,
Normal/Inverted direction, and reset-to-raw all behave correctly and never mutate
`raw_depth`; the 16-bit PNG export genuinely writes 16-bit data (65,536 distinct levels
verified on a gradient, not silently truncated to 8-bit's 256); and DA3Engine's
non-inference logic — the per-checkpoint license table, capability flags, and its graceful
fallback when torch isn't installed — all check out against the compatibility report's own
numbers. Run `python tests/test_pipeline.py` (or `pytest tests/`) yourself to reproduce.

## Two corrections vs. the report you provided

1. **Marigold's dtype kwarg.** The report showed `dtype=torch.float16`. Real usage examples
   from the diffusers ecosystem use `torch_dtype=torch.float16`. `torch_dtype` is the safe
   choice across diffusers versions; `dtype` may be a newer alias on very recent releases.
   Not wired into any code yet (Marigold is Milestone 2) — just flagging it before you get
   there so it isn't copied verbatim.
2. **Marigold checkpoint licensing isn't uniform across versions.** v1-1 (the one this
   project targets) is confirmed `openrail++` on its HF model card, matching the report. But
   `marigold-depth-v1-0` was separately *re-licensed to Apache-2.0* back in Dec 2023 — worth
   knowing once the Model Manager (spec section 15) can list more than one Marigold
   checkpoint, since they won't all be under the same terms.

Everything else in the report's DA3/Marigold API shapes and license claims held up against
primary sources.

## Setup

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3 && pip install -e . && cd ..
pip install -r requirements.txt
python main.py
```

First click of "Generate Depth" downloads the `da3mono-large` checkpoint from the Hugging
Face Hub via `from_pretrained` if it isn't cached yet. Spec section 15 wants model downloads
to be explicit and user-controlled; Milestone 1 has no Model Manager yet, so for now that
download just happens on first use. Worth fixing when the Model Manager is built.

## One open question, and a script to settle it

`app/engines/da3/engine.py` doesn't assume whether `prediction.depth` comes back sized to
`process_res` (504px) or already resampled to your source image — nothing I could find
states this either way (see the module's docstring). The adapter checks the returned shape
and resizes if needed, so it's correct regardless, but you don't actually know which branch
is running. Run this on your GPU machine:

```bash
python quick_check.py
```

It prints `source_resolution` next to `raw_depth.shape` for a deliberately non-square,
non-504 test image. If you share the result, the adapter's comments (and the resize check
itself, if it turns out to be dead code) can be simplified accordingly.

## Structure

```
depth-studio/
├── main.py                       entry point
├── quick_check.py                resolution-verification script (see above)
├── requirements.txt
├── tests/
│   └── test_pipeline.py          runtime-tested, no GPU needed — see above
├── app/
│   ├── core/
│   │   ├── image_io.py           load source image, preserve it untouched (spec §4)
│   │   └── depth_data.py         RAW → NORMALIZED → DISPLAY tiers, Normal/Inverted (spec §5, §6)
│   ├── engines/
│   │   ├── base.py               DepthEngine interface + shared dataclasses (spec §2)
│   │   └── da3/engine.py         DA3Engine adapter
│   ├── export/
│   │   └── depth_png16.py        16-bit PNG export, never the colorized preview (spec §12)
│   └── ui/
│       └── main_window.py        minimal PySide6 window (spec §17 layout, simplified)
```

## Next (Milestone 2, per spec section 24)

Marigold engine + `MarigoldEngine` adapter, an Engine Manager so the UI isn't hardcoded to
DA3, a real Model Manager (license-per-row, explicit downloads — spec §15), model comparison
(spec §18), and the refinement controls (spec §8). None of that is started here on purpose —
this milestone is deliberately just the one path through the pipeline, end to end.
