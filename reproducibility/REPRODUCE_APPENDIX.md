# Reproduce Appendix

This note covers the appendix figures, the broadened benchmark-view assets, and the final hardening-round appendix tables.

## 1. Regenerate the descriptor ablation used by the appendix implementation section

```bash
python scripts/build_descriptor_ablation.py \
  --skip-rwtd-final \
  --descriptor-modes handcrafted ptd_convnext hybrid_ptd \
  --out-root results/artifacts/descriptor_ablation
```

This command writes:

- `results/artifacts/descriptor_ablation/descriptor_ablation_summary.json`
- `results/artifacts/descriptor_ablation/descriptor_ablation_summary.csv`

The RWTD rows intentionally stop at Stage-A: the dense rescue layer is descriptor-agnostic and is unchanged across descriptor variants, so the ablation isolates the descriptor-sensitive commitment layer only.

## 2. Practicality table inputs

The practicality table is assembled from retained profiling and audit artifacts rather than a single builder script. See:

- `results/UPDATED_SOURCES_USED.md`
- `results/RESULTS_MANIFEST.md`

## 3. Regenerate the feature-space summary figure

```bash
python paper/scripts/build_feature_summary_figure.py
```

This command regenerates:

- `paper/figures/fig_feature_space_recovery.png`

from the archived parity CSVs and the retained RWTD flip-holdout gallery assets.

## 4. Regenerate the appendix figures built in this round

```bash
python paper/scripts/build_appendix_assets.py
```

This command regenerates:

- `paper/figures/fig_feature_gallery.png`
- `paper/figures/fig_feature_scale_diagnostics.png`
- `paper/figures/fig_rwtd_case_gallery.png`
- `paper/figures/fig_stld_case_gallery.png`
- local copies of:
  - `paper/figures/fig_detexture_audit.png`
  - `paper/figures/fig_controlnet_bridge_examples.png`
  - `paper/figures/fig_caid_audit.png`

## 5. Regenerate the broader-complement figures

Run these before `paper/scripts/build_appendix_assets.py` if you want the manuscript tree to pick up refreshed benchmark-complement figure labels.

### ADE20K-selected audit figure

```bash
python scripts/build_detexture_audit_figure.py
```

### ControlNet bridge audit figure

```bash
python scripts/build_controlnet_audit_figure.py
```

### CAID audit figure

```bash
python scripts/build_caid_audit_figure.py
```

### Four-route composite audit used in the main paper

```bash
python scripts/build_neurips_explanatory_figures.py
```

## 6. Rebuild the PDF after regenerating appendix assets

```bash
cd paper
tectonic main.tex
```
