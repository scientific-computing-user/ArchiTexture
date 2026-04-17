# Reproduce Appendix

This note covers the appendix-only figures and supporting route views added in the packaging round, plus the final hardening-round appendix tables.

## 1. Regenerate the descriptor ablation used by the appendix implementation section

```bash
conda run -n openworldsam python /home/galoren/ArchiTexture/TextureSAM-v2/scripts/build_descriptor_ablation.py \
  --skip-rwtd-final \
  --descriptor-modes handcrafted ptd_convnext hybrid_ptd \
  --out-root /home/galoren/ArchiTexture/TextureSAM-v2/reports/final_submission_descriptor_ablation
```

This command writes:

- `TextureSAM-v2/reports/final_submission_descriptor_ablation/descriptor_ablation_summary.json`
- `TextureSAM-v2/reports/final_submission_descriptor_ablation/descriptor_ablation_summary.csv`
- per-mode RWTD summaries under `.../<mode>/rwtd/`
- per-mode STLD summaries under `.../<mode>/stld/`

The RWTD rows intentionally stop at Stage-A: the dense rescue layer is descriptor-agnostic and is unchanged across descriptor variants, so the hardening-round ablation isolates the descriptor-sensitive commitment layer only.

## 2. Practicality table inputs

The practicality table is assembled from retained profiling and audit artifacts rather than a single builder script:

- RWTD Stage-A counts:
  - `TextureSAM-v2/reports/margin_guard_sweep_swin256/margin_0p0/summary.json`
- RWTD dense-rescue counts:
  - `TextureSAM-v2/reports/release_swinb_full256_audit/texturesam2_acute_learned_rescue/per_image.csv`
- RWTD retained runtime profile:
  - `TextureSAM-v2/reports/camera_ready_reviewer_mitigation/runtime_profile/v9_timing.json`
  - `TextureSAM-v2/reports/camera_ready_reviewer_mitigation/runtime_profile/v11_timing.json`
- STLD Stage-A counts:
  - `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/eval/strict_ptd_learned/summary.json`

## 3. Regenerate the feature-space summary figure

```bash
python /home/galoren/ArchiTexture/paper_neurips_unified/scripts/build_feature_summary_figure.py
```

This command regenerates:

- `paper_neurips_unified/figures/fig_feature_space_recovery.png`

from the archived SAM2-vs-SAM3 parity CSVs and the retained RWTD flip-holdout gallery assets.

## 4. Regenerate the appendix figures built in this round

```bash
python /home/galoren/ArchiTexture/paper_neurips_unified/scripts/build_appendix_assets.py
```

This command regenerates:

- `paper_neurips_unified/figures/fig_feature_gallery.png`
- `paper_neurips_unified/figures/fig_feature_scale_diagnostics.png`
- `paper_neurips_unified/figures/fig_rwtd_case_gallery.png`
- `paper_neurips_unified/figures/fig_stld_case_gallery.png`
- local copies of:
  - `paper_neurips_unified/figures/fig_controlnet_bridge_examples.png`
  - `paper_neurips_unified/figures/fig_caid_audit.png`

The builder now copies the supporting ControlNet and CAID figures from the refreshed `TextureSAM-v2/TextureSum2_paper/figures/` outputs rather than from the older `paper_main` stash.

## 5. Source bundles used by the appendix builder

- Feature-space qualitative previews:
  - `research-site/site/experiments/sam2-vs-sam3-frozen-feature-clustering/assets/all_previews`
- Feature-space scale diagnostics:
  - `research-site/site/experiments/sam3-stage2-coarse-vs-fine-probe/assets/tables/summary_table.csv`
- RWTD selector / oracle cases:
  - `TextureSAM-v2/reports/reviewer_oracles_rwtd_full256/rwtd_proposal_oracles_per_image.csv`
  - `TextureSAM-v2/reports/final_round_learned_single_selector_rwtd/rwtd_learned_single_selector_per_image.csv`
  - `TextureSAM-v2/reports/release_swinb_full256_audit/texturesam2_acute_learned_rescue/diagnostics/audit_cases`
- STLD selector / oracle cases:
  - `TextureSAM-v2/reports/reviewer_oracles_stld/stld_proposal_oracles_per_image.csv`
  - `TextureSAM-v2/reports/final_round_learned_single_selector_stld/stld_learned_single_selector_per_image.csv`
  - `TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/benchmark`

## 6. Regenerate the supporting-route figures from their original scripts

Run these before `build_appendix_assets.py` if you want the manuscript tree to pick up refreshed supporting-route figure labels.

### ControlNet bridge

```bash
python /home/galoren/ArchiTexture/TextureSAM-v2/scripts/build_controlnet_bridge_figure.py
```

### CAID audit figure

```bash
python /home/galoren/ArchiTexture/TextureSAM-v2/scripts/build_caid_audit_figure.py
```

## 7. Supporting deck note

The supporting deck audit remains available at [SUPPORTING_DECK_AUDIT.md](/home/galoren/ArchiTexture/SUPPORTING_DECK_AUDIT.md). This hardening round does not add new slide-derived figures; it keeps the appendix focused on the already in-scope feature-space diagnostics and proposal-space supporting evidence.

## 8. Rebuild the PDF after regenerating appendix assets

```bash
cd /home/galoren/ArchiTexture/paper_neurips_unified
tectonic main.tex
```
