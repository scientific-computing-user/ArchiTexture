# RWTD Miner

Resumable Python pipeline for mining RWTD-like texture-transition images from dense segmentation datasets.

This repo includes:
- The full mining codebase (Stage A/B/C/D, adapters, scoring, reporting, review UI).
- A **GitHub Pages-ready static review site** under `docs/review/` (full exported bundle).

## Features

- Disk-budget batching (`--batch_budget_gb`) so each batch stays within storage limits.
- Multi-stage narrowing pipeline:
  - Stage A: mask-statistics prefilter (multiprocess).
  - Stage B: CLIP/SigLIP text-image retrieval (open_clip preferred, transformers fallback).
  - Stage C: optional caption keyword filtering.
  - Stage D: optional pluggable VLM scoring backend.
- Crash-safe checkpoints per batch.
- Per-batch and merged manifests (Parquet + CSV).
- `selected/` exports by symlink (or copy fallback).
- QA artifacts: contact sheets, histograms, stage summaries.
- Static review website per batch with sortable gallery and detail panel:
  - original image
  - annotation mask visualization
  - texture-boundary overlay
  - selection rationale and stage metrics

## Hosted Review (Public Site)

This code repository stays private.
The public static review website is published separately at:

- `https://scientific-computing-user.github.io/rwtd-texture-miner-site/review/`
- Public site repo: `https://github.com/scientific-computing-user/rwtd-texture-miner-site`

Local preview:
- Open `docs/review/index.html` directly.

The bundled site in `docs/review` is fully static (no backend) and includes:
- thumbnails
- original images
- mask visualizations
- texture-boundary overlays
- sortable/filterable gallery and detail panel

## Folder Conventions (Local adapter)

Expected default layout under `input_root`:

- `images/` -> `.jpg/.jpeg/.png/.webp`
- `annotations/` -> `.json` (one file per image or shard JSONs)

The adapter supports:

1. One JSON per image (`annotations/<image_id>.json`)
2. Shard JSONs (`annotations/*.json`) with common SA-1B-like schemas (best effort).

## Install

```bash
cd /Users/galoren/TextureData/rwtd_miner_github_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Profile-Based Runs (Recommended)

Create merged runtime config from base + profile:

```bash
python scripts/merge_config.py \
  --base config.yaml \
  --profile configs/profiles/imac_cpu.yaml \
  --out .tmp/config_imac_cpu.yaml
```

Server-ready bootstrap:

```bash
bash scripts/bootstrap_env.sh --profile server_rtx3090_fast
```

One-command bootstrap + run:

```bash
bash scripts/bootstrap_and_run.sh \
  --profile server_rtx3090_fast \
  --out /path/to/ade20k_eval \
  --skip_download
```

Run ADE20K with CLIP + VLM using a profile:

```bash
bash scripts/run_ade20k_rwtd.sh \
  --profile server_rtx3090_fast \
  --out /path/to/ade20k_eval \
  --skip_download
```

Migration + handoff docs:
- `SERVER_MIGRATION.md`
- `CODEX_SERVER_HANDOFF.md`
- `SERVER_NON_ADE20K_SCOPE.md` (next-dataset scope, time estimates, server-only download plan)

Resume context quickly (useful after moving to another machine):

```bash
bash scripts/codex_resume.sh
```

Plan/download the non-ADE20K dataset scope on server:

```bash
bash scripts/prepare_non_ade20k_server.sh --root /data/rwtd_datasets --mode plan
bash scripts/prepare_non_ade20k_server.sh --root /data/rwtd_datasets --mode download_public
```

## CLI

Build index:

```bash
python -m rwtd_miner.cli index \
  --input_root /path/to/sa1b \
  --out /path/to/out \
  --config /Users/galoren/TextureData/rwtd_miner/config.yaml
```

Run one batch:

```bash
python -m rwtd_miner.cli run \
  --input_root /path/to/sa1b \
  --out /path/to/out \
  --batch 0 \
  --batch_budget_gb 5
```

Run first N batches:

```bash
python -m rwtd_miner.cli run \
  --input_root /path/to/sa1b \
  --out /path/to/out \
  --max_batches 10 \
  --batch_budget_gb 5
```

Dry run (Stage A + B only):

```bash
python -m rwtd_miner.cli run \
  --input_root /path/to/sa1b \
  --out /path/to/out \
  --skip_vlm
```

Pilot 100 samples from SA-1B shard + review website:

```bash
python -m rwtd_miner.cli pilot100 \
  --out /path/to/pilot_out \
  --num_images 100 \
  --batch_budget_gb 0.8 \
  --skip_vlm
```

Calibrate thresholds by combining SA-1B pilot with the original RWTD (`Kaust256`) from TextureSAM:

```bash
python -m rwtd_miner.cli calibrate_rwtd \
  --pilot_out /path/to/pilot_out \
  --out /path/to/calibration_out \
  --max_rwtd_images 256 \
  --selected_min 60 \
  --borderline_min 50 \
  --sync_html_to /Users/galoren/TextureData/outputs/html_review
```

Run full ADE20K test and count images passing the RWTD-calibrated minimum bar:

```bash
python -m rwtd_miner.cli ade20k_full \
  --out /path/to/ade20k_eval \
  --selected_min 60 \
  --borderline_min 50 \
  --review_limit 1500 \
  --sync_html_to /Users/galoren/TextureData/outputs/html_review
```

Run ADE20K with CLIP + VLM (local BLIP VQA backend):

```bash
python -m rwtd_miner.cli ade20k_full \
  --out /path/to/ade20k_eval_multimodal \
  --skip_download \
  --enable_clip \
  --enable_vlm \
  --vlm_backend hf_blip_vqa \
  --vlm_top_n 700 \
  --selected_min 60 \
  --borderline_min 50
```

Run ADE20K with CLIP + VLM (external command backend):

```bash
export GEMINI_API_KEY="YOUR_KEY"
python -m rwtd_miner.cli ade20k_full \
  --out /path/to/ade20k_eval_multimodal \
  --skip_download \
  --enable_clip \
  --enable_vlm \
  --vlm_backend external_command \
  --vlm_external_command "python /absolute/path/scripts/vlm_wrapper_gemini.py --model gemini-2.5-flash-lite" \
  --vlm_top_n 700 \
  --selected_min 60 \
  --borderline_min 50
```

## Output Structure

`out/`

- `index/`
  - `image_index.parquet`
  - `batch_assignments.parquet`
- `batches/batch_<id>/`
  - `batch_manifest.parquet`
  - `batch_manifest.csv`
  - `checkpoint.json`
  - `selected/`
  - `borderline/`
  - `review/`
    - `index.html`
    - `data.json`
    - `thumbnails/`
    - `masks/`
    - `overlays/`
  - `qa/`
    - `summary.json`
    - `score_histogram.png`
    - `contact_sheet_top100.jpg`
    - `stageB_pass_sample_grid.jpg`
    - `stageB_fail_sample_grid.jpg`
- `manifests/`
  - `all_processed.parquet`
  - `all_processed.csv`

## GitHub Pages Workflow

This repo ships `.github/workflows/pages.yml`:
- builds Pages artifact from `docs/`
- deploys to GitHub Pages on push to `main`

## Notes

- CPU-only is default. If CUDA/MPS is available and enabled in config, Stage B/C can use it.
- Stage D is pluggable. Supported backends: `stub`, `hf_blip_vqa`, `hf_vlm_chat`, `external_command`.
- Missing/invalid annotations are handled gracefully (`stageA_status=unknown`) with optional CLIP fallback.

## Free Remote VLM (low local compute)

If you want Stage D to stay fast/light locally, use remote inference via `external_command`.

Included wrapper:

- `scripts/vlm_wrapper_gemini.py`

It reads `{"image_path": "...", "prompt": "..."}` from stdin and returns Stage-D JSON to stdout.

Setup:

```bash
cd /Users/galoren/TextureData/rwtd_miner
source .venv/bin/activate
export GEMINI_API_KEY="YOUR_KEY"
```

Quick sanity test of wrapper contract:

```bash
echo '{"image_path":"/abs/path/sample.jpg","prompt":"Score this image and return JSON only."}' | \
python /Users/galoren/TextureData/rwtd_miner/scripts/vlm_wrapper_gemini.py --model gemini-2.5-flash-lite
```

Use in config (`stage_d`):

```yaml
stage_d:
  enabled: true
  backend: external_command
  external_command: "python /Users/galoren/TextureData/rwtd_miner/scripts/vlm_wrapper_gemini.py --model gemini-2.5-flash-lite"
  score_top_n_from_stage_b: 700
```

Then run normally (without `--skip_vlm`).

## Troubleshooting

- If open_clip is unavailable, the code automatically falls back to transformers CLIP.
- If pycocotools is unavailable, Stage A still works with annotation `area` fields and partial RLE fallback.
- If Parquet dependencies are missing, CSV output still works.
