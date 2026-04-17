# ArchiTexture

ArchiTexture studies texture segmentation in frozen SAM-style systems as a recoverability problem. The main paper claim is narrow: strong texture performance can be recovered by learning proposal-space commitment above a frozen proposal bank, without retraining the backbone or proposal generator.

This public repo packages the final paper, the `texturesam_v2` implementation used by the proposal-space route, the in-scope experiment scripts, and the retained result artifacts behind the final manuscript tables and appendix diagnostics.

## Scope

- Main-paper benchmarks: RWTD and STLD
- Diagnostic only: feature-space recovery from coarse frozen features
- Appendix only: ControlNet bridge and CAID
- Excluded from this release's main story: DeTexture / Detector / ADE20K and AdaSam-style adaptor experiments

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests
```

## Paper And Results

- Paper PDF: [paper/main.pdf](paper/main.pdf)
- Paper source and build notes: [paper/README.md](paper/README.md)
- Main result and appendix reproduction notes: [reproducibility/](reproducibility)
- Figure/table traceability: [results/RESULTS_MANIFEST.md](results/RESULTS_MANIFEST.md)
- Exact retained experiment commands: [results/EXPERIMENT_LEDGER.md](results/EXPERIMENT_LEDGER.md)
- Final hardening-round summary: [results/FINAL_SUBMISSION_RESULTS.md](results/FINAL_SUBMISSION_RESULTS.md)

## What Is Frozen vs Learned

- Frozen:
  - the SAM-style proposal generator and proposal bank
  - the backbone used by the auxiliary feature probe
- Learned:
  - proposal compatibility scoring
  - conservative component scoring and selection
  - the RWTD rescue layer

## Repo Layout

- `texturesam_v2/`: core proposal-space implementation package
- `scripts/`: curated in-scope experiment, evaluation, and analysis scripts
- `tests/`: lightweight package tests
- `paper/`: final NeurIPS manuscript source and compiled PDF
- `appendix_assets/`: standalone appendix galleries and supporting figures
- `reproducibility/`: relative-path reproduction notes for main-paper and appendix artifacts
- `results/`: retained summary JSON/CSV files plus manifests and ledgers
- `data_docs/`: benchmark-role notes
- `checkpoints_manifest/`: external checkpoint expectations for full reruns
- `docs/`: release scope notes

## Reproducibility Note

This repo ships the retained summary artifacts behind the published tables and figures. Full route reruns still require local benchmark roots, SAM upstream assets, and pretrained checkpoints that are not committed here. The fastest audit path is to verify the committed summaries in `results/artifacts/` and rebuild the paper from `paper/`.
