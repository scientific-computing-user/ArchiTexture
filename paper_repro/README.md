# Paper Table Reproduction

This directory makes the paper tables reproducible from the public `ArchiTexture` checkout itself.

## Quick Start

Rebuild and verify the committed paper tables:

```bash
python paper_repro/scripts/build_paper_tables.py --check --write
```

That command does two things:

- Rebuilds the paper tables from the committed summary artifacts under `paper_repro/source_data/`.
- Fails if any rebuilt row drifts from the table values committed in `paper_repro/table_manifest.json`.

The rendered markdown bundle is written to `paper_repro/generated/paper_tables.md`.

## What Is In Scope

This bundle covers the paper-facing tables that matter for reproduction:

- Main paper protocol summary
- Main four-route results table
- RWTD hypothesis-isolation baselines
- RWTD proposal-source swap table
- Appendix route-secondary tables for ControlNet-stitched PTD and CAID
- Appendix compact reproducibility summary
- Appendix RWTD failure-audit summary
- Appendix RWTD control and proposal-swap tables
- Appendix RWTD confidence summary

Each table is mapped in `paper_repro/table_manifest.json` to:

- the public source file(s) committed in this repo
- the original local workspace file(s) they came from
- the rebuild mode used by `build_paper_tables.py`

## Reproducibility Model

This public bundle is intentionally split into two tiers.

`Tier 1: GitHub-only table reproduction`

- Works from this repo alone.
- Uses compact committed JSON/CSV summaries in `paper_repro/source_data/`.
- Lets a reader regenerate the exact reported table rows without rerunning heavyweight model inference.

`Tier 2: Full experiment reruns`

- Requires the companion experiment workspace documented in `../TextureSAM-v2`.
- Requires the upstream evaluator / SAM2 checkout documented in `../TextureSAM_upstream_20260303`.
- Route-specific scripts and archived output roots are documented in [PAPER_REPRODUCIBILITY.md](../PAPER_REPRODUCIBILITY.md).

This split is deliberate: the GitHub repo now guarantees table-level reproducibility, while the heavier end-to-end reruns still depend on external datasets, checkpoints, and archived experiment workspaces.

## Important Note On The RWTD Confidence Table

The RWTD confidence table is included here as a final-manuscript export in `paper_repro/source_data/manuscript_exports/rwtd_confidence_summary.json`.

The means and effective overlap counts are backed by committed public source files, but the exact instance-level bootstrap export used for the final CI endpoints does not appear to have been preserved as a standalone JSON in the companion workspace. The public bundle states that explicitly instead of pretending otherwise.
