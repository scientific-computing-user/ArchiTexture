# Paper

This directory contains the manuscript source and compiled PDF for:

**Does Frozen SAM Understand Texture? Evidence from Frozen Features and Proposal Masks**

The paper is organized around two parallel evidence routes:

- **Feature-space exploration** over frozen multiscale features
- **Proposal-/mask-space exploration** over generated proposal masks

## Compile

```bash
cd paper
tectonic main.tex
```

This produces `paper/main.pdf`.

## Contents

- `main.tex`, `abstract.tex`, `sections/`, `tables/`, `figures/`: manuscript source
- `appendix.tex`: appendix included after the references in the same PDF
- `main.pdf`: prebuilt compiled manuscript
- `scripts/build_appendix_assets.py`: appendix-asset builder retained from the paper round

The source is self-contained for the retained manuscript build.
