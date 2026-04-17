# ArchiTexture

<p align="center">
  <strong>Does Frozen SAM Understand Texture? Evidence from Frozen Features and Proposal Masks</strong>
</p>

<p align="center">
  Paper, code, and retained artifacts for a question-led NeurIPS study of texture understanding in frozen SAM-style systems.
</p>

<p align="center">
  <a href="paper/main.pdf"><strong>Paper PDF</strong></a>
  ·
  <a href="paper/main.tex"><strong>LaTeX Source</strong></a>
  ·
  <a href="reproducibility/REPRODUCE_MAIN_RESULTS.md"><strong>Reproduce Main Results</strong></a>
  ·
  <a href="reproducibility/REPRODUCE_APPENDIX.md"><strong>Reproduce Appendix</strong></a>
  ·
  <a href="results/RESULTS_MANIFEST.md"><strong>Results Manifest</strong></a>
</p>

## The Question

This repository accompanies a narrow scientific question:

**Does frozen SAM already contain texture understanding, even though it was built mainly for semantic segmentation?**

The paper answers that question through two complementary routes:

1. **Feature-space exploration**
   - What texture structure is already latent in frozen multiscale features?
2. **Proposal-/mask-space exploration**
   - What texture structure is already present in the generated mask bank, and how much of it is recoverable without retraining the backbone or proposal generator?

The repo name remains `ArchiTexture`, but the paper’s identity is the question above rather than an implementation label.

## Two Evidence Routes

| Route | Frozen evidence inspected | What it reveals | What it does not settle alone |
| --- | --- | --- | --- |
| **Feature-space exploration** | multiscale SAM `backbone_fpn` features | latent texture-relevant organization in the frozen representation | matched end-task recoverability under the official RWTD / STLD comparison blocks |
| **Proposal-/mask-space exploration** | generated proposal masks | recoverable texture partitions already externalized into the mask bank | whether the texture signal was already present before proposal generation |

Read together, these routes argue that frozen SAM is not texture-blind: texture-relevant structure is visible both in the representation and in the generated masks.

## Main Matched Findings

Values are reported as `mIoU / ARI`.

| Benchmark | Evaluator / subset | Comparator | Proposal route | Reading |
| --- | --- | --- | --- | --- |
| RWTD | official invariant, matched shared subset | TextureSAM rerun `0.4684 / 0.6163` | **`0.4645 / 0.7013`** | near-matched overlap, much stronger coherence |
| RWTD | official invariant, full evaluator | SAM2.1-small rerun `0.1615 / 0.2183` | **`0.4611 / 0.6966`** | large gain over the raw frozen baseline |
| STLD | direct foreground, matched shared subset | TextureSAM rerun `0.5140 / 0.7526` | **`0.7195 / 0.7791`** | stronger overlap and better coherence |
| STLD | direct foreground, all images | SAM2.1-small rerun `0.3686 / 0.5269` | **`0.6705 / 0.7249`** | large gain in both views |

The feature-space route is reported separately under its archived probe protocol and is used to show that coarse frozen features already organize nontrivial texture structure.

## Why The Paper Is Interesting

- **It separates latent evidence from recoverable evidence.** Frozen features and generated masks answer different parts of the same scientific question.
- **It explains why benchmarks differ.** RWTD behaves like a fragmented-evidence regime, while STLD often behaves like a singleton-selection regime.
- **It shows that ambiguity, not total signal absence, is the remaining bottleneck.** RWTD oracle analysis reveals substantial unrecovered headroom in the frozen mask bank.
- **It stays narrow and disciplined.** The central main-body benchmarks are RWTD and STLD; bridge and CAID routes remain supporting appendix material.

## What Stays Frozen

| Frozen | Learned above it |
| --- | --- |
| SAM-style backbone and proposal generator | feature probe clustering controls |
| generated proposal mask bank | proposal compatibility scoring |
| prompt source / proposal generation process | conservative component scoring and selection |
| backbone weights | RWTD-only rescue readout |

The result is a clean recoverability study rather than a new end-to-end adaptation stack.

## Repository Tour

| Path | Purpose |
| --- | --- |
| `paper/` | final manuscript source and compiled PDF |
| `texturesam_v2/` | core proposal-/mask-space package |
| `scripts/` | in-scope experiment, evaluation, and analysis scripts |
| `tests/` | lightweight package tests |
| `results/` | retained summaries, manifests, and experiment ledger |
| `appendix_assets/` | standalone appendix figures and galleries |
| `reproducibility/` | shortest-path notes for rebuilding main and appendix artifacts |
| `docs/` | release-scope and support notes |
| `data_docs/` | benchmark-role notes |
| `checkpoints_manifest/` | expectations for external checkpoints needed for full reruns |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m unittest discover -s tests -v
```

To rebuild the paper PDF:

```bash
cd paper
tectonic main.tex
```

## Reproducibility Path

1. Read the paper: [paper/main.pdf](paper/main.pdf)
2. Map every figure and table to its artifact source: [results/RESULTS_MANIFEST.md](results/RESULTS_MANIFEST.md)
3. Inspect retained command provenance: [results/EXPERIMENT_LEDGER.md](results/EXPERIMENT_LEDGER.md)
4. Use [reproducibility/REPRODUCE_MAIN_RESULTS.md](reproducibility/REPRODUCE_MAIN_RESULTS.md) for the shortest path back to the matched tables
5. Use [reproducibility/REPRODUCE_APPENDIX.md](reproducibility/REPRODUCE_APPENDIX.md) for the appendix galleries and supporting figures

## Scope Discipline

This public package follows the final paper scope:

- **Main paper:** RWTD and STLD
- **Parallel scientific routes:** feature-space exploration and proposal-/mask-space exploration
- **Supporting appendix routes:** ControlNet bridge and CAID
- **Out of scope for the main story:** DeTexture / Detector / ADE20K and AdaSam-style adaptor branches

## Citation

If you use this repository, please cite the paper and repository metadata in [CITATION.cff](CITATION.cff).
