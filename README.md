# ArchiTexture

<p align="center">
  <strong>Does Frozen SAM Understand Texture? Evidence from Frozen Features and Proposal Masks</strong>
</p>

<p align="center">
  Paper, code, retained artifacts, and reproducibility notes for a question-led NeurIPS study of texture understanding in frozen SAM-style models.
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

This repository accompanies one narrow scientific question:

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
| **Feature-space exploration** | multiscale SAM `backbone_fpn` features | latent texture-relevant organization in the frozen representation | matched end-task recoverability under the route-specific benchmark evaluators |
| **Proposal-/mask-space exploration** | generated proposal masks | recoverable texture partitions already externalized into the frozen mask bank | whether that structure was already visible before proposal generation |

Read together, these routes argue that frozen SAM is not texture-blind: texture-relevant structure is visible both in the representation and in the generated masks.

## Benchmark Picture

The paper does not stop at one historical natural benchmark and one historical synthetic benchmark. It keeps the gold-standard anchors, then broadens each side with a more current complement.

| Role | Route | Why it is here |
| --- | --- | --- |
| **Historical natural anchor** | RWTD | hard fragmented natural texture scenes; strongest test of commitment above a frozen bank |
| **Natural complement** | ADE20K-selected texture crops | visually selected public natural-image subset that broadens the natural side beyond the older small benchmark |
| **Historical synthetic anchor** | STLD | canonical synthetic shaped-foreground benchmark with exact supervision |
| **Synthetic complement** | ControlNet bridge | exact synthetic labels with more naturalistic stitched transitions than classical mosaics |
| **Appendix real-world complement** | CAID | shoreline / overhead-imagery binary partitions with exact masks and unusually large contiguous regions |

The first four routes sit in the main paper’s proposal-/mask-space story. CAID is retained separately in the appendix because it is informative but more domain-specific.

## Main Matched Findings

Values are reported as `mIoU / ARI`. Evaluator semantics are route-specific and are documented in the paper and in [data_docs/README.md](data_docs/README.md).

| Benchmark | Evaluator / subset | Comparator | Proposal route | Reading |
| --- | --- | --- | --- | --- |
| RWTD | official invariant, matched shared subset | TextureSAM rerun `0.4684 / 0.6163` | **`0.4645 / 0.7013`** | near-matched overlap, much stronger coherence |
| RWTD | official invariant, full evaluator | SAM2.1-small rerun `0.1615 / 0.2183` | **`0.4611 / 0.6966`** | large gain over the raw frozen baseline |
| ADE20K-selected | invariant, matched validation subset | TextureSAM rerun `0.4715 / 0.3306` | **`0.5043 / 0.3762`** | broader natural-image complement keeps the same frozen-bank reading |
| STLD | direct foreground, matched shared subset | TextureSAM rerun `0.5140 / 0.7526` | **`0.7195 / 0.7791`** | stronger overlap and better coherence |
| STLD | direct foreground, all images | SAM2.1-small rerun `0.3686 / 0.5269` | **`0.6705 / 0.7249`** | large gain in both views |
| ControlNet bridge | invariant, matched shared subset | TextureSAM rerun `0.6436 / 0.5519` | **`0.6803 / 0.6039`** | more naturalistic synthetic seams still benefit from learned commitment |

The feature-space route is reported separately under its archived probe protocol and is used to show that coarse frozen features already organize nontrivial texture structure.

## Why This Paper Is Interesting

- **It separates latent evidence from recoverable evidence.** Frozen features and generated masks answer different parts of the same question.
- **It broadens the benchmark picture instead of leaning only on older anchors.** ADE20K-selected crops complement RWTD on the natural side, and the ControlNet bridge complements STLD on the synthetic side.
- **It explains why benchmark regimes differ.** RWTD behaves like a fragmented-evidence problem, while STLD often behaves like a singleton-selection problem.
- **It shows that ambiguity, not total signal absence, is the remaining bottleneck.** RWTD oracle analysis reveals substantial unrecovered headroom in the frozen mask bank.
- **It stays narrow and disciplined.** The main paper keeps four central benchmark routes and one appendix-only real-world complement rather than reopening older diffuse branches.

## What Stays Frozen

| Frozen | Learned above it |
| --- | --- |
| SAM-style backbone and proposal generator | feature probe clustering controls |
| generated proposal mask bank | proposal compatibility scoring |
| prompt source / proposal generation process | conservative component scoring and selection |
| backbone weights | RWTD-only rescue readout |

The result is a recoverability study, not a new end-to-end adaptation stack.

## Repository Tour

| Path | Purpose |
| --- | --- |
| `paper/` | final manuscript source and compiled PDF |
| `texturesam_v2/` | core proposal-/mask-space package |
| `scripts/` | in-scope experiment, evaluation, and figure-construction scripts |
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
2. Map every figure and table to its source artifact: [results/RESULTS_MANIFEST.md](results/RESULTS_MANIFEST.md)
3. Inspect retained command provenance: [results/EXPERIMENT_LEDGER.md](results/EXPERIMENT_LEDGER.md)
4. Use [reproducibility/REPRODUCE_MAIN_RESULTS.md](reproducibility/REPRODUCE_MAIN_RESULTS.md) for the shortest path back to the main matched tables
5. Use [reproducibility/REPRODUCE_APPENDIX.md](reproducibility/REPRODUCE_APPENDIX.md) for the appendix galleries and the complement figures

## Scope Discipline

This public package follows the final paper scope:

- **Question-led main paper:** frozen-feature evidence plus frozen-mask evidence
- **Historical anchors:** RWTD and STLD
- **Broader complements:** ADE20K-selected natural crops and the ControlNet bridge synthetic route
- **Appendix real-world complement:** CAID
- **Out of scope for the main story:** Detector-style legacy branches and AdaSam-style adaptor experiments

## Citation

If you use this repository, please cite the paper and repository metadata in [CITATION.cff](CITATION.cff).
