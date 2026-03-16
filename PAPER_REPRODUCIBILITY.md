# ArchiTexture Paper Reproducibility Map

This file makes the current local artifact layout explicit.

## Scope

The important thing to understand is that the local paper workspace used for the manuscript is **not identical** to this GitHub repo checkout.

- This repo: public entrypoint, project overview, mining/review code, bundled static review site.
- Companion paper workspace in this local archive: `../TextureSAM-v2`
- Companion manuscript package: `../TextureSAM-v2/TextureSum2_paper`
- Official evaluator and upstream SAM2 code: `../TextureSAM_upstream_20260303`
- Public gallery repo/worktree: `../rwtd_miner_public_site`

If you only read one file after `README.md`, read this one.

## GitHub-Only Table Reproduction

The public repo now carries a paper-table bundle under [`paper_repro/`](paper_repro/).

- Entry point: [paper_repro/README.md](paper_repro/README.md)
- Rendered tables: [paper_repro/generated/paper_tables.md](paper_repro/generated/paper_tables.md)
- Machine-readable map: `paper_repro/table_manifest.json`
- Verifier: `python paper_repro/scripts/build_paper_tables.py --check --write`

This bundle is the answer to the narrow question, "Can I reproduce the table values from the GitHub repo itself?" It works from committed summary artifacts and flags drift against the final paper rows. The remainder of this file covers the heavier companion-workspace routes used to regenerate those summaries from archived experiment outputs.

## Workspace Roots

| Local path | Role |
| --- | --- |
| `../TextureSAM-v2` | Main paper experiment workspace |
| `../TextureSAM-v2/TextureSum2_paper` | Manuscript, figures, tables, Overleaf exports, release JSON/CSV artifacts |
| `../TextureSAM_upstream_20260303` | Official `eval_no_agg_masks.py` and upstream `sam2/` |
| `../TextureSAM_upstream_assets` | Checkpoints used for reproduced TextureSAM/SAM2 inference |
| `../rwtd_miner_public_site` | Public result galleries referenced by the paper |
| `./rwtd_miner` | Mining/review pipeline retained in this repo |

## RWTD: Main Natural Route

### Core source files

- `../TextureSAM-v2/scripts/run_official_texturesam_inference.py`
- `../TextureSAM-v2/scripts/build_promptstyle_multibank.py`
- `../TextureSAM-v2/scripts/run_strict_ptd_v3.py`
- `../TextureSAM-v2/scripts/run_strict_ptd_v8_partition.py`
- `../TextureSAM-v2/scripts/run_strict_ptd_v7_dualgate.py`
- `../TextureSAM-v2/scripts/run_strict_ptd_v9_v7_v8_gate.py`
- `../TextureSAM-v2/scripts/run_strict_ptd_v11_dense_rescue.py`
- `../TextureSAM-v2/scripts/export_single_masks_as_official.py`
- `../TextureSAM-v2/scripts/eval_upstream_texture_metrics.py`

### Key archived output roots

- `../TextureSAM-v2/reports/repro_upstream_eval`
  Reproduced TextureSAM official rerun and official-format masks.
- `../TextureSAM-v2/reports/strict_ptd_v9_v7v8_on_official0p3`
  Conservative v9 gate outputs.
- `../TextureSAM-v2/reports/strict_ptd_v11_multibank`
  Dense multi-bank v11 route outputs.
- `../TextureSAM-v2/reports/texturesum2_official_noagg_eval`
  Archived official-format exports and official no-agg JSONs for paper-side evaluation.
- `../TextureSAM-v2/reports/release_swinb_full256_audit`
  Released full-256 ArchiTexture export, audit tables, top-k follow-up, margin mixes.
- `../TextureSAM-v2/reports/release_swinb_pre_ring_non_tweaked`
  Core/pre-repair official exports used for denominator-controlled appendix comparisons.
- `../TextureSAM-v2/reports/reviewer_upgrade_baselines_full256`
  Hypothesis-isolation proposal-bank baselines.

### Paper-critical summary builders

- `../TextureSAM-v2/scripts/build_strong_accept_audit.py`
  Writes:
  - `../TextureSAM-v2/TextureSum2_paper/data/strong_accept_audit.json`
  - `../TextureSAM-v2/TextureSum2_paper/data/coverage_slice_table.csv`
  - `../TextureSAM-v2/TextureSum2_paper/figures/fig_targeted_rescue_bins.png`
  - `../TextureSAM-v2/TextureSum2_paper/figures/fig_win_tie_loss.png`
- `../TextureSAM-v2/scripts/build_e2e_subset_report.py`
  Writes the end-to-end subset sanity-check protocol/results/timing.
- `../TextureSAM-v2/scripts/build_reviewer_mitigation_pack.py`
  Writes baseline-integrity, robustness, runtime, component ablation, and the leakage lock manifest.
- `../TextureSAM-v2/scripts/build_release_appendix_tables.py`
  Writes denominator fairness, paired significance, practical abstention, and cross-dataset appendix tables.
- `../TextureSAM-v2/scripts/build_reviewer_followup_pack.py`
  Writes the common-253 follow-up policy artifacts and associated figures.
- `../TextureSAM-v2/scripts/build_rwtd_difficulty_breakdown.py`
  Writes the GT-geometry difficulty slices.

### Minimal official reevaluation commands

```bash
python ../TextureSAM-v2/scripts/eval_upstream_texture_metrics.py \
  --pred-folder ../TextureSAM-v2/reports/release_swinb_full256_audit/official_export \
  --gt-folder ../TextureSAM_upstream_20260303/Kaust256/labeles \
  --upstream-root ../TextureSAM_upstream_20260303 \
  --out-json ../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_full256.json
```

```bash
python ../TextureSAM-v2/scripts/eval_upstream_texture_metrics.py \
  --pred-folder ../TextureSAM-v2/reports/release_swinb_full256_audit/official_export_common253 \
  --gt-folder ../TextureSAM_upstream_20260303/Kaust256/labeles \
  --upstream-root ../TextureSAM_upstream_20260303 \
  --out-json ../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_common253.json
```

## STLD: Controlled Synthetic Route

### Experiment root

- `../TextureSAM-v2/experiments/khan_synthetic_gallery_20260312`

### Main entrypoints

- `../TextureSAM-v2/scripts/build_khan_synthetic_benchmark.py`
- `../TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/run_gallery.sh`
- `../TextureSAM-v2/scripts/eval_stld_direct.py`
- `../TextureSAM-v2/scripts/publish_khan_synthetic_gallery.py`
- `../TextureSAM-v2/scripts/build_stld_audit_figure.py`

### Key outputs

- `benchmark/`
  Held-out STLD-style synthetic images and labels.
- `infer_official_0p3_p24/`
  Reproduced TextureSAM mask bank.
- `promptstyle/`
  Prompt-style bank converted from the official-format rerun.
- `union_masks/`
  Proposal-union baseline.
- `eval/strict_ptd_learned/`
  ArchiTexture masks and per-image diagnostics.
- `direct_eval/direct_foreground_summary.json`
  Direct foreground numbers used in the paper.

## ControlNet-Stitched PTD: Bridge Route

### Experiment root

- `../TextureSAM-v2/experiments/perlin_controlnet_eval_20260312`

### Main entrypoints

- `../TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/run_stagea_hi_eval.sh`
- `../TextureSAM-v2/scripts/run_official_texturesam_inference.py`
- `../TextureSAM-v2/scripts/build_promptstyle_multibank.py`
- `../TextureSAM-v2/scripts/run_stagea_learned_only.py`
- `../TextureSAM-v2/scripts/eval_binary_partition_maskbank.py`
- `../TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/run_gallery.sh`
- `../TextureSAM-v2/scripts/publish_controlnet_stitched_gallery.py`
- `../TextureSAM-v2/scripts/build_controlnet_audit_figure.py`
- `../TextureSAM-v2/scripts/build_controlnet_bridge_figure.py`

### Key outputs

- `full_0p3/infer_0p3/`
  Reproduced TextureSAM mask bank.
- `full_0p3/promptstyle_0p3/`
  Prompt-style bank.
- `full_0p3/stageA_0p3/strict_ptd_learned/masks`
  ArchiTexture Stage-A outputs.
- `full_0p3/eval_0p3/binary_partition_summary.json`
  Partition-invariant and direct metrics.

## CAID: Domain-Specific Shoreline Route

### Experiment root

- `../TextureSAM-v2/experiments/caid_eval_20260313`

### Main entrypoints

- `../TextureSAM-v2/scripts/build_caid_binary_benchmark.py`
- `../TextureSAM-v2/experiments/caid_eval_20260313/run_stagea_eval.sh`
- `../TextureSAM-v2/scripts/run_official_texturesam_inference.py`
- `../TextureSAM-v2/scripts/build_promptstyle_multibank.py`
- `../TextureSAM-v2/scripts/run_stagea_learned_only.py`
- `../TextureSAM-v2/scripts/eval_binary_partition_maskbank.py`
- `../TextureSAM-v2/scripts/publish_caid_gallery.py`
- `../TextureSAM-v2/scripts/build_caid_audit_figure.py`

### Key outputs

- `benchmarks/caid_test/`
  Local binary shoreline benchmark.
- `full_0p3/infer_0p3/`
  Reproduced TextureSAM mask bank.
- `full_0p3/stageA_0p3/strict_ptd_learned/masks`
  ArchiTexture Stage-A outputs.
- `full_0p3/eval_0p3/binary_partition_summary.json`
  Partition-invariant and direct metrics.

## Figure Builders Used By The Manuscript

### Main paper figures

- `../TextureSAM-v2/scripts/build_neurips_explanatory_figures.py`
  Generates:
  - `fig_ptd_rwtd_bridge.png`
  - `fig_consolidation_logic.png`
  - `fig_route_audits.png`
  - and related explanatory figures in `../TextureSAM-v2/TextureSum2_paper/figures/`

### Route-specific audit figures

- `../TextureSAM-v2/scripts/build_stld_audit_figure.py`
- `../TextureSAM-v2/scripts/build_controlnet_audit_figure.py`
- `../TextureSAM-v2/scripts/build_caid_audit_figure.py`

### Supplement/reviewer figures

- `../TextureSAM-v2/scripts/build_strong_accept_audit.py`
- `../TextureSAM-v2/scripts/build_reviewer_mitigation_pack.py`
- `../TextureSAM-v2/scripts/build_reviewer_followup_pack.py`
- `../TextureSAM-v2/scripts/build_rwtd_difficulty_breakdown.py`

## Manuscript Package

The manuscript-facing source of truth lives in:

- `../TextureSAM-v2/TextureSum2_paper/main.tex`
- `../TextureSAM-v2/TextureSum2_paper/paper_body.tex`
- `../TextureSAM-v2/TextureSum2_paper/appendix_body.tex`
- `../TextureSAM-v2/TextureSum2_paper/neurips_supplement.tex`
- `../TextureSAM-v2/TextureSum2_paper/REPRO_COMMANDS.md`
- `../TextureSAM-v2/TextureSum2_paper/SUPPLEMENT_CONFIG_TABLE.md`
- `../TextureSAM-v2/TextureSum2_paper/OVERLEAF_TRANSFER.md`

Useful exported bundles already present in the companion workspace include:

- `../TextureSAM-v2/TextureSum2_paper/ArchiTexture_overleaf_projects_20260312.zip`
- `../TextureSAM-v2/TextureSum2_paper/ArchiTexture_neurips_main_overleaf.zip`
- `../TextureSAM-v2/TextureSum2_paper/ArchiTexture_neurips_supplement_overleaf.zip`

## Public Artifact Builders

The paper also depends on the public result galleries, which are generated from the companion workspace and published in the separate public-site repo:

- `../TextureSAM-v2/scripts/build_texturesam2_ai_docs.py`
- `../TextureSAM-v2/scripts/publish_rwtd_gallery_variant.py`
- `../TextureSAM-v2/scripts/publish_khan_synthetic_gallery.py`
- `../TextureSAM-v2/scripts/publish_controlnet_stitched_gallery.py`
- `../TextureSAM-v2/scripts/publish_caid_gallery.py`

Published outputs live in:

- `../rwtd_miner_public_site/docs/texturesam2_ai_gallery`
- `../rwtd_miner_public_site/docs/khan_synthetic_gallery`
- `../rwtd_miner_public_site/docs/controlnet_stitched_gallery`
- `../rwtd_miner_public_site/docs/caid_gallery`

## Why This File Exists

The current `ArchiTexture` GitHub repo name is correct, but the local paper artifact was produced from a companion workspace with a denser experiment history than this public repo currently exposes. This file exists to remove ambiguity: if you need to know which code, which reports, which figures, and which archived outputs produced the paper, this is the map.
