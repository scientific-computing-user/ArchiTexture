# ArchiTexture Paper Tables

Generated from `paper_repro/table_manifest.json` and `paper_repro/source_data/`.

Rebuild with `python paper_repro/scripts/build_paper_tables.py --check --write`.

## Main Paper

### `tab:route_protocol`

Per-route protocol summary

Static protocol table from the final manuscript. The public repo keeps the final row values here and maps them back to the paper-side config documents.

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/paper_body.tex`
- `../TextureSAM-v2/TextureSum2_paper/SUPPLEMENT_CONFIG_TABLE.md`
- `../TextureSAM-v2/TextureSum2_paper/REPRO_COMMANDS.md`

| Route | Train source | Deployed variant | Primary metric | Comparator |
| --- | --- | --- | --- | --- |
| RWTD | PTD synthetic supervision only | Full ArchiTexture: Swin-Base encoder + merge/core/repair | Official RWTD mIoU / ARI | prior paper-reported state of the art + public checkpoint rerun + raw SAM2.1-small |
| STLD | Brodatz + MPEG-7 synthetic supervision | Full ArchiTexture matched to STLD construction | Direct foreground mIoU / ARI | public checkpoint rerun + raw SAM2.1-small |
| ControlNet-stitched PTD | none beyond PTD training | PTD-trained `small_pre_ring_hi` Stage-A | Partition-invariant mIoU / ARI | public checkpoint rerun + raw SAM2.1-small |
| CAID | none beyond PTD training | PTD-trained `small_pre_ring_hi` Stage-A | Partition-invariant mIoU / ARI | public checkpoint rerun + raw SAM2.1-small |

### `tab:results`

Unified results across the four evaluation routes

This table is rebuilt directly from the committed summary JSON files. The paper-reported TextureSAM RWTD row remains a contextual manuscript number rather than a recomputed public-checkpoint rerun.

Public sources:
- `source_data/main_results/rwtd_sam2_original_official.json`
- `source_data/main_results/rwtd_texturesam_common253_official.json`
- `source_data/main_results/rwtd_architexture_full256_official.json`
- `source_data/main_results/rwtd_architexture_common253_official.json`
- `source_data/main_results/stld_sam2_original_summary.json`
- `source_data/main_results/stld_architexture_summary.json`
- `source_data/main_results/stld_texturesam_summary.json`
- `source_data/main_results/controlnet_sam2_original_summary.json`
- `source_data/main_results/controlnet_partition_summary.json`
- `source_data/main_results/caid_sam2_original_summary.json`
- `source_data/main_results/caid_partition_summary.json`

Original workspace sources:
- `../TextureSAM-v2/experiments/sam2_original_baselines_20260313/rwtd/sam2_original_rwtd_eval.json`
- `../TextureSAM-v2/reports/repro_upstream_eval/official_0p3_full_combined_upstream_eval.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_full256.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_common253.json`
- `../TextureSAM-v2/experiments/sam2_original_baselines_20260313/stld/sam2_original_stld_summary.json`
- `../TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/direct_eval/direct_foreground_summary.json`
- `../TextureSAM-v2/experiments/khan_synthetic_gallery_20260312/stld_texturesam0p3_maskbank_summary.json`
- `../TextureSAM-v2/experiments/sam2_original_baselines_20260313/controlnet/sam2_original_controlnet_summary.json`
- `../TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/full_0p3/eval_0p3/binary_partition_summary.json`
- `../TextureSAM-v2/experiments/sam2_original_baselines_20260313/caid/sam2_original_caid_summary.json`
- `../TextureSAM-v2/experiments/caid_eval_20260313/full_0p3/eval_0p3/binary_partition_summary.json`

| Method | Subset / coverage | mIoU | ARI |
| --- | --- | --- | --- |
| **RWTD (natural route)** |  |  |  |
| SAM2.1-small original rerun | full-256, 256/256 | 0.1615 | 0.2183 |
| TextureSAM paper-reported RWTD result (eta <= 0.3) | paper-reported | 0.4700 | 0.6200 |
| ArchiTexture (ours) | full-256 | 0.4611 | 0.6966 |
| TextureSAM public checkpoint rerun | common-253 | 0.4684 | 0.6163 |
| ArchiTexture (ours) | common-253 | 0.4645 | 0.7013 |
| **STLD (controlled synthetic route)** |  |  |  |
| SAM2.1-small original rerun | all-200, 199/200 | 0.3686 | 0.5269 |
| TextureSAM public checkpoint rerun | all-200, 182/200 | 0.4677 | 0.6849 |
| ArchiTexture (ours) | all-200, 200/200 | 0.6705 | 0.7249 |
| TextureSAM public checkpoint rerun | common-182 | 0.5140 | 0.7526 |
| ArchiTexture (ours) | common-182 | 0.7195 | 0.7791 |
| **ControlNet-stitched PTD (generator-based bridge route)** |  |  |  |
| SAM2.1-small original rerun | all-1742, 1742/1742 | 0.2770 | 0.2213 |
| TextureSAM public checkpoint rerun | all-1742, 1739/1742 | 0.6425 | 0.5510 |
| ArchiTexture Stage-A (ours) | all-1742, 1742/1742 | 0.6803 | 0.6039 |
| **CAID (domain-specific shoreline route)** |  |  |  |
| SAM2.1-small original rerun | all-3104, 3104/3104 | 0.2588 | 0.2040 |
| TextureSAM public checkpoint rerun | all-3104, 3063/3104 | 0.6691 | 0.5080 |
| ArchiTexture Stage-A (ours) | all-3104, 3104/3104 | 0.6677 | 0.5745 |

### `tab:hypothesis_baselines`

RWTD hypothesis-isolation baselines under the same proposal-bank protocol

The six baseline rows come from the proposal-bank baseline summary; the final row comes from the released full-256 official evaluator rerun.

Public sources:
- `source_data/rwtd_controls/release_proposal_bank_baselines_summary.csv`
- `source_data/main_results/rwtd_architexture_full256_official.json`

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/data/release_proposal_bank_baselines_summary.csv`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_full256.json`

| Method | mIoU | ARI |
| --- | --- | --- |
| Core only | 0.4543 | 0.6786 |
| Repair only | 0.4835 | 0.6101 |
| Rank only | 0.4536 | 0.4305 |
| Union | 0.4699 | 0.1622 |
| Intersection | 0.2733 | 0.2972 |
| Lightweight reranker | 0.4004 | 0.3629 |
| ArchiTexture final | 0.4611 | 0.6966 |

### `tab:proposal_swap`

Frozen-source transfer stress test

This table is rebuilt from the release proposal-source swap CSV.

Public sources:
- `source_data/rwtd_controls/release_proposal_source_swap.csv`

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/data/release_proposal_source_swap.csv`

| Dense rescue proposal source (frozen) | Full-256 mIoU / ARI | Common-253 mIoU / ARI |
| --- | --- | --- |
| Released dense bank | 0.4611 / 0.6966 | 0.4645 / 0.7013 |
| SAM2.1-large source swap | 0.5003 / 0.7048 | 0.5005 / 0.7106 |

## Appendix

### `app:controlnet_secondary`

Secondary ControlNet-stitched PTD view

This table is rebuilt from the committed ControlNet partition summary.

Public sources:
- `source_data/main_results/controlnet_partition_summary.json`

Original workspace sources:
- `../TextureSAM-v2/experiments/perlin_controlnet_eval_20260312/full_0p3/eval_0p3/binary_partition_summary.json`

| Method | Subset / coverage | Direct mIoU / ARI | Invariant mIoU / ARI |
| --- | --- | --- | --- |
| TextureSAM 0.3 released checkpoint | all-1742, 1739/1742 | 0.4212 / 0.6446 | 0.6425 / 0.5510 |
| ArchiTexture Stage-A (ours) | all-1742, 1742/1742 | 0.4993 / 0.6039 | 0.6803 / 0.6039 |
| TextureSAM 0.3 released checkpoint | common-1739 | 0.4219 / 0.6457 | 0.6436 / 0.5519 |
| ArchiTexture Stage-A (ours) | common-1739 | 0.4993 / 0.6039 | 0.6803 / 0.6039 |

### `app:caid_secondary`

CAID under the same partition-invariant protocol used in the main paper

This table is rebuilt from the committed CAID partition summary.

Public sources:
- `source_data/main_results/caid_partition_summary.json`

Original workspace sources:
- `../TextureSAM-v2/experiments/caid_eval_20260313/full_0p3/eval_0p3/binary_partition_summary.json`

| Method | Subset / coverage | Invariant mIoU / ARI |
| --- | --- | --- |
| TextureSAM public checkpoint rerun | all-3104, 3063/3104 | 0.6691 / 0.5080 |
| ArchiTexture Stage-A (ours) | all-3104, 3104/3104 | 0.6677 / 0.5745 |
| TextureSAM public checkpoint rerun | common-3063 | 0.6780 / 0.5148 |
| ArchiTexture Stage-A (ours) | common-3063 | 0.6677 / 0.5745 |

### `app:compact_repro`

Compact reproducibility summary for the configurations used in the paper

Static reproducibility summary from the final appendix table.

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/appendix_body.tex`
- `../TextureSAM-v2/TextureSum2_paper/SUPPLEMENT_CONFIG_TABLE.md`

| Item | Value |
| --- | --- |
| External training data | PTD only for the RWTD route; Brodatz-native synthetic supervision for STLD |
| Encoder training data | 80,000 train / 20,000 val images per synthetic source |
| RWTD benchmark usage | Evaluation only; no RWTD-label training or hyperparameter search |
| Proposal source | Frozen SAM-style automatic masks |
| RWTD/STLD deployed model | Swin-Base masked-region encoder + learned merge/core/repair modules |
| ControlNet / CAID deployed model | PTD-trained `small_pre_ring_hi` Stage-A bundle |
| Primary evaluator | Official upstream no-aggregation evaluator on RWTD; route-matched evaluators elsewhere |
| Comparator family | Raw SAM2.1-small and reproduced/public TextureSAM checkpoints |

### `app:failure_audit`

Full-256 failure-audit summary for the released RWTD decision rule

This row is rebuilt from the full-256 audit summaries plus the committed per-image release CSV for top-k recoverability.

Public sources:
- `source_data/rwtd_audit/audit_summary.json`
- `source_data/rwtd_audit/rescue_reliability_summary.json`
- `source_data/rwtd_audit/wrong_partition_summary.json`
- `source_data/rwtd_audit/release_per_image.csv`

Original workspace sources:
- `../TextureSAM-v2/reports/release_swinb_full256_audit/audit/audit_summary.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/audit/rescue_reliability_summary.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/audit/wrong_partition_summary.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/texturesam2_acute_learned_rescue/per_image.csv`

| Setting | Resc. | Off-succ. | Hurt-any | Wrong | Unsafe | Mean DeltaIoU | Top-k recov. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline decision rule | 21 | 61.9% | 38.1% | 4 | 6 | +0.2247 | 141/256 |

### `app:rwtd_controls`

RWTD hypothesis-isolation baselines under the same proposal-bank protocol

The appendix view extends the main hypothesis-isolation table with common-253 numbers and interpretive behavior labels.

Public sources:
- `source_data/rwtd_controls/release_proposal_bank_baselines_summary.csv`
- `source_data/main_results/rwtd_architexture_full256_official.json`
- `source_data/main_results/rwtd_architexture_common253_official.json`

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/data/release_proposal_bank_baselines_summary.csv`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_full256.json`
- `../TextureSAM-v2/reports/release_swinb_full256_audit/official_eval_common253.json`

| Method | Full-256 mIoU / ARI | Common-253 mIoU / ARI | Key behavior |
| --- | --- | --- | --- |
| Compatibility-only core | 0.4543 / 0.6786 | 0.4558 / 0.6812 | conservative, consistency-preserving |
| Repair-only | 0.4835 / 0.6101 | 0.4876 / 0.6148 | coverage-seeking, consistency drop |
| Ranking-only | 0.4536 / 0.4305 | 0.4559 / 0.4333 | severe partition errors |
| Union heuristic | 0.4699 / 0.1622 | 0.4706 / 0.1622 | over-merge collapse |
| Intersection heuristic | 0.2733 / 0.2972 | 0.2766 / 0.3007 | under-coverage collapse |
| Lightweight reranker | 0.4004 / 0.3629 | 0.4027 / 0.3642 | unstable tradeoff |
| ArchiTexture final | 0.4611 / 0.6966 | 0.4645 / 0.7013 | best consistency with competitive coverage |

### `app:proposal_swap_detail`

Frozen-source transfer stress test with appendix notes

Appendix version of the proposal-source swap table with the release note column preserved.

Public sources:
- `source_data/rwtd_controls/release_proposal_source_swap.csv`

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/data/release_proposal_source_swap.csv`

| Dense rescue proposal source (frozen) | Full-256 mIoU / ARI | Common-253 mIoU / ARI | Notes |
| --- | --- | --- | --- |
| Released dense bank (official0.3 setup) | 0.4611 / 0.6966 | 0.4645 / 0.7013 | Main-paper release numbers |
| Source swap: SAM2.1-large bank | 0.5003 / 0.7048 | 0.5005 / 0.7106 | Same frozen consolidation weights; only dense rescue candidates change |

### `app:confidence_summary`

Official-score confidence summary for the RWTD comparisons reported in the paper

The CI endpoints come from the final manuscript export. See the export note in source_data/manuscript_exports/rwtd_confidence_summary.json for why this table is not fully recomputed from raw bootstrap samples inside the public repo.

Public sources:
- `source_data/manuscript_exports/rwtd_confidence_summary.json`

Original workspace sources:
- `../TextureSAM-v2/TextureSum2_paper/appendix_body.tex`
- `../TextureSAM-v2/TextureSum2_paper/neurips_supplement.tex`

| Setting | Subset | mIoU | 95% CI (mIoU) | ARI | 95% CI (ARI) | n_eff |
| --- | --- | --- | --- | --- | --- | --- |
| ArchiTexture final | full-256 | 0.4611 | [0.4196, 0.5027] | 0.6966 | [0.6627, 0.7299] | 432 |
| ArchiTexture final | common-253 | 0.4645 | [0.4230, 0.5060] | 0.7013 | [0.6672, 0.7350] | 426 |
| ArchiTexture core-only | full-256 | 0.4543 | [0.4130, 0.4956] | 0.6786 | [0.6423, 0.7135] | 428 |
| TextureSAM public rerun | common-253 | 0.4684 | [0.4427, 0.4948] | 0.6163 | [0.5892, 0.6431] | 493 |
