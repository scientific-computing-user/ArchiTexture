#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== RWTD Miner Resume Context ==="
echo "repo_root: $ROOT_DIR"
echo "hostname: $(hostname)"
echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

printf "\n[git]\n"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  echo "branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "commit: $(git rev-parse --short HEAD)"
  echo "status:"
  git status --short | head -n 40
} || echo "not a git repository"

printf "\n[env]\n"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -V
  python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count())
PY
else
  echo ".venv not found"
fi

printf "\n[profiles]\n"
ls -1 configs/profiles/*.yaml 2>/dev/null || true

printf "\n[handoff docs]\n"
for f in \
  SERVER_START_HERE.md \
  SERVER_MIGRATION.md \
  SERVER_NON_ADE20K_SCOPE.md \
  CODEX_SERVER_HANDOFF.md \
  README.md; do
  [[ -f "$f" ]] && echo "$f"
done

printf "\n[dataset registry]\n"
if [[ -f configs/datasets/non_ade20k_registry.yaml ]]; then
  echo "configs/datasets/non_ade20k_registry.yaml"
  head -n 80 configs/datasets/non_ade20k_registry.yaml
else
  echo "registry not found"
fi

printf "\n[latest summaries]\n"
find . -type f \( -name 'ade20k_eval_summary*.json' -o -name 'calibration_summary.json' -o -name 'bundle_summary.json' \) -print | head -n 20

printf "\n[github push health]\n"
echo "auth setup script: scripts/setup_server_github_auth.sh"
echo "push check script: scripts/check_server_github_push.sh"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "origin: $(git remote get-url origin 2>/dev/null || true)"
fi

printf "\n[next suggested command]\n"
echo "bash scripts/run_ade20k_rwtd.sh --profile server_rtx3090_fast --out /path/to/output --skip_download"

printf "\n[codex bootstrap prompt]\n"
cat <<'EOF'
Read SERVER_START_HERE.md, SERVER_NON_ADE20K_SCOPE.md, and configs/datasets/non_ade20k_registry.yaml.
Then inspect current adapters and implement/execute the highest-priority pending dataset pipeline with VLM enabled, resumable checkpoints, and review-site outputs.
EOF
