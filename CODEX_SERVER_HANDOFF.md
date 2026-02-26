# Codex Handoff (When opening Codex on server)

## First command to run in terminal

```bash
cd /home/<user>/rwtd_miner_github_repo && bash scripts/codex_resume.sh
```

This prints repository status, environment status, available profiles, handoff docs, dataset registry, latest summaries, and a bootstrap prompt for Codex.

## Recommended first prompt to Codex on server

"Run `bash scripts/codex_resume.sh`, then read `SERVER_START_HERE.md`, `SERVER_NON_ADE20K_SCOPE.md`, and `configs/datasets/non_ade20k_registry.yaml`. Continue with VLM-enabled server execution and resumable outputs."

## One-command execution after bootstrap

```bash
bash scripts/run_ade20k_rwtd.sh --profile server_rtx3090_fast --out /data/rwtd_runs/ade20k_fast --skip_download
```

## Files Codex should inspect first
- `SERVER_START_HERE.md`
- `SERVER_NON_ADE20K_SCOPE.md`
- `configs/datasets/non_ade20k_registry.yaml`
- `config.yaml`
- `configs/profiles/server_rtx3090_fast.yaml`
- `configs/profiles/server_rtx3090_quality.yaml`
- `rwtd_miner/stages/stage_b_clip.py`
- `rwtd_miner/stages/stage_d_vlm.py`
- latest manifest under `<out>/manifests/`
