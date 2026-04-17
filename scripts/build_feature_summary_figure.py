#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PAPER_ROOT.parent
EXPERIMENT_ROOT = (
    WORKSPACE_ROOT
    / "research-site"
    / "site"
    / "experiments"
    / "sam2-vs-sam3-frozen-feature-clustering"
)
COARSE_CSV = EXPERIMENT_ROOT / "assets" / "tables" / "coarse_only_model_parity.csv"
FLIP_CSV = EXPERIMENT_ROOT / "assets" / "tables" / "flip_averaged_model_parity.csv"
TRAINING_DATA = EXPERIMENT_ROOT / "training_data.json"
OUT_PATH = PAPER_ROOT / "figures" / "fig_feature_space_recovery.png"


SAM2_COARSE = "#6f89bd"
SAM3_COARSE = "#e39a56"
SAM2_FLIP = "#3b66b7"
SAM3_FLIP = "#e1730f"


def _load_holdout_panels() -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(TRAINING_DATA.read_text())
    story = next(item for item in data["stories"] if item["story_id"] == "rwtd_flip_holdout")
    by_label = {item["label"]: item for item in story["images"]}

    sam2_path = EXPERIMENT_ROOT / by_label["SAM2 flip"]["asset_rel"]
    sam3_path = EXPERIMENT_ROOT / by_label["SAM3 flip"]["asset_rel"]
    return np.asarray(Image.open(sam2_path).convert("RGB")), np.asarray(Image.open(sam3_path).convert("RGB"))


def main() -> None:
    coarse = pd.read_csv(COARSE_CSV).set_index("dataset_label").loc[["RWTD", "CAID", "STLD"]]
    flip = pd.read_csv(FLIP_CSV).set_index("dataset_label").loc[["RWTD", "CAID", "STLD"]]
    sam2_img, sam3_img = _load_holdout_panels()

    fig = plt.figure(figsize=(13.6, 8.2), dpi=220)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.28, wspace=0.10)

    ax_miou = fig.add_subplot(gs[0, 0])
    ax_ari = fig.add_subplot(gs[0, 1])
    ax_sam2 = fig.add_subplot(gs[1, 0])
    ax_sam3 = fig.add_subplot(gs[1, 1])

    datasets = coarse.index.tolist()
    x = np.arange(len(datasets))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    bar_specs = [
        ("SAM2 coarse-only", coarse["sam2_miou"].to_numpy(), coarse["sam2_ari"].to_numpy(), SAM2_COARSE, None),
        ("SAM3 coarse-only", coarse["sam3_miou"].to_numpy(), coarse["sam3_ari"].to_numpy(), SAM3_COARSE, None),
        ("SAM2 flip-avg", flip["sam2_miou"].to_numpy(), flip["sam2_ari"].to_numpy(), SAM2_FLIP, "//"),
        ("SAM3 flip-avg", flip["sam3_miou"].to_numpy(), flip["sam3_ari"].to_numpy(), SAM3_FLIP, "//"),
    ]

    for idx, (label, miou_vals, ari_vals, color, hatch) in enumerate(bar_specs):
        kw = dict(width=width, label=label, color=color, edgecolor=color)
        if hatch is not None:
            kw["hatch"] = hatch
            kw["edgecolor"] = "#1f2937"
            kw["linewidth"] = 0.6
        ax_miou.bar(x + offsets[idx], miou_vals, **kw)
        ax_ari.bar(x + offsets[idx], ari_vals, **kw)

    for ax, metric in [(ax_miou, "mIoU"), (ax_ari, "ARI")]:
        ax.set_xticks(x, datasets)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.22)

    ax_miou.set_title("A. Coarse pooling already reveals strong partitions", fontsize=12.5)
    ax_ari.set_title("B. Flip-averaging is a positional debiasing aid, not a universal winner", fontsize=12.5)
    ax_ari.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax_sam2.imshow(sam2_img)
    ax_sam2.set_title("C. RWTD flip-avg holdout: SAM2 view", fontsize=12.5)
    ax_sam2.axis("off")

    ax_sam3.imshow(sam3_img)
    ax_sam3.set_title("D. Same RWTD holdout: SAM3 view", fontsize=12.5)
    ax_sam3.axis("off")

    fig.suptitle("Feature-space exploration of frozen SAM backbones", fontsize=16.5, y=0.985)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
