#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from texturesam_v2.ptd_data import PTDImageBackend, group_entries_by_class, load_ptd_entries


FIG_DIR = ROOT / "TextureSum2_paper" / "figures"
RWTD_IMG_DIR = Path("/home/galoren/TextureSAM_upstream_20260303/Kaust256/images_textured")
RWTD_GT_DIR = Path("/home/galoren/TextureSAM_upstream_20260303/Kaust256/labeles")
PTD_ROOT = Path("/home/galoren/repo/data/ptd")
BASELINE_DIR = ROOT / "reports/repro_upstream_eval/official_0p3_masks"
OURS_DIR = ROOT / "reports/release_swinb_full256_audit/official_export"
CORE_DIR = ROOT / "reports/margin_guard_sweep_swin256/margin_0p0/masks"
DENSE_BANK_DIR = ROOT / "reports/sam21large_official_masks"
AUDIT_CASES_DIR = ROOT / "reports/release_swinb_full256_audit/texturesam2_acute_learned_rescue/diagnostics/audit_cases"
FAILURE_AUDIT_CSV = ROOT / "reports/release_swinb_full256_audit/audit/per_image_failure_audit.csv"
PER_IMAGE_CSV = ROOT / "reports/release_swinb_full256_audit/texturesam2_acute_learned_rescue/per_image.csv"


@dataclass(frozen=True)
class FactorSpec:
    image_id: int
    title: str
    note: str


def _draw_text(ax, text: str) -> None:
    ax.text(
        0.02,
        0.03,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", fc=(0, 0, 0, 0.56), ec="none"),
    )


def _center_crop_resize(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    src_ratio = w / max(h, 1)
    dst_ratio = out_w / max(out_h, 1)
    if src_ratio > dst_ratio:
        new_w = int(h * dst_ratio)
        x0 = max((w - new_w) // 2, 0)
        crop = image[:, x0 : x0 + new_w]
    else:
        new_h = int(w / max(dst_ratio, 1e-8))
        y0 = max((h - new_h) // 2, 0)
        crop = image[y0 : y0 + new_h, :]
    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _center_crop_resize_pair(image: np.ndarray, mask: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    src_ratio = w / max(h, 1)
    dst_ratio = out_w / max(out_h, 1)
    if src_ratio > dst_ratio:
        new_w = int(h * dst_ratio)
        x0 = max((w - new_w) // 2, 0)
        image_crop = image[:, x0 : x0 + new_w]
        mask_crop = mask[:, x0 : x0 + new_w]
    else:
        new_h = int(w / max(dst_ratio, 1e-8))
        y0 = max((h - new_h) // 2, 0)
        image_crop = image[y0 : y0 + new_h, :]
        mask_crop = mask[y0 : y0 + new_h, :]
    image_out = cv2.resize(image_crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_out = cv2.resize(mask_crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return image_out, mask_out


def _load_rwtd_overlay(image_id: int, width: int = 520, height: int = 360) -> np.ndarray:
    img = cv2.imread(str(RWTD_IMG_DIR / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(RWTD_GT_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise FileNotFoundError(f"RWTD example missing for image id {image_id}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, gt = _center_crop_resize_pair(img, gt, width, height)

    # Draw all partition boundaries to avoid cherry-picking a single partition.
    boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    out = img.copy()
    out[boundary] = np.array([45, 190, 70], dtype=np.uint8)
    return out


def _load_rwtd_pair(image_id: int, width: int = 520, height: int = 360) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(RWTD_IMG_DIR / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(RWTD_GT_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise FileNotFoundError(f"RWTD example missing for image id {image_id}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, gt = _center_crop_resize_pair(img, gt, width, height)
    return img, gt


def _load_rwtd_pair_no_crop(image_id: int, width: int = 520, height: int = 360) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(RWTD_IMG_DIR / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(RWTD_GT_DIR / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
    if img is None or gt is None:
        raise FileNotFoundError(f"RWTD example missing for image id {image_id}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
    return img, gt


def _find_masks(root: Path, image_id: int, pattern: str) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in sorted(root.glob(pattern.format(image_id=image_id))):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out.append(m > 0)
    return out


def _overlay_masks(image: np.ndarray, masks: list[np.ndarray], colors: list[tuple[int, int, int]], alpha: float = 0.25) -> np.ndarray:
    out = image.copy().astype(np.float32)
    for idx, mask in enumerate(masks):
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        c = np.array(colors[idx % len(colors)], dtype=np.float32).reshape(1, 1, 3)
        out[mask] = 0.72 * out[mask] + alpha * c
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colors[idx % len(colors)], 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_boundary(image: np.ndarray, labels: np.ndarray, color: tuple[int, int, int] = (45, 190, 70)) -> np.ndarray:
    out = image.copy()
    b = cv2.morphologyEx(labels.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
    out[b] = np.array(color, dtype=np.uint8)
    return out


def _overlay_single_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (240, 140, 35), alpha: float = 0.28) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = image.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = 0.72 * out[mask] + alpha * c
    out = np.clip(out, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 3)
    return out


def build_pipeline_clear_figure() -> None:
    image_id = 15  # User-selected diagnostic case.
    case_dir = AUDIT_CASES_DIR / str(image_id)

    def read_rgb(name: str) -> np.ndarray:
        p = case_dir / name
        arr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def read_mask(name: str) -> np.ndarray:
        p = case_dir / name
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise FileNotFoundError(p)
        return arr > 0

    out_w, out_h = 420, 310
    img = cv2.resize(read_rgb("image_rgb.png"), (out_w, out_h), interpolation=cv2.INTER_AREA)
    old_overlay = cv2.resize(read_rgb("candidate_stack_overlay.png"), (out_w, out_h), interpolation=cv2.INTER_AREA)
    support_heat = cv2.resize(read_rgb("proposal_support_heatmap.png"), (out_w, out_h), interpolation=cv2.INTER_AREA)
    diff_final_gt = cv2.resize(read_rgb("diff_final_vs_gt.png"), (out_w, out_h), interpolation=cv2.INTER_AREA)
    gt_mask = cv2.resize(read_mask("gt_partition_mask.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    baseline_mask = cv2.resize(read_mask("baseline_scored_mask.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    core_mask = cv2.resize(read_mask("core_mask.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    best_mask = cv2.resize(read_mask("best_ranked_candidate_mask.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    final_mask = cv2.resize(read_mask("final_selected_mask.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
    overlap_union_mask = cv2.resize(read_mask("candidate_overlap_union.png").astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0

    panel_input = _overlay_boundary(img, gt_mask.astype(np.uint8))
    panel_gt = _overlay_boundary(_overlay_mask(img, gt_mask, (56, 190, 88), alpha=0.28, thickness=2), gt_mask.astype(np.uint8))
    panel_baseline = _overlay_boundary(_overlay_mask(img, baseline_mask, (149, 117, 205), alpha=0.32, thickness=2), gt_mask.astype(np.uint8))
    panel_core = _overlay_boundary(_overlay_mask(img, core_mask, (66, 133, 244), alpha=0.33, thickness=2), gt_mask.astype(np.uint8))
    panel_best = _overlay_boundary(_overlay_mask(img, best_mask, (245, 194, 66), alpha=0.31, thickness=2), gt_mask.astype(np.uint8))
    panel_final = _overlay_boundary(_overlay_mask(img, final_mask, (244, 140, 35), alpha=0.33, thickness=2), gt_mask.astype(np.uint8))
    panel_overlay = _overlay_boundary(old_overlay, gt_mask.astype(np.uint8))
    panel_union = _overlay_boundary(_overlay_mask(img, overlap_union_mask, (221, 83, 70), alpha=0.30, thickness=2), gt_mask.astype(np.uint8))
    panel_support = _overlay_boundary(support_heat, gt_mask.astype(np.uint8))
    panel_diff = _overlay_boundary(diff_final_gt, gt_mask.astype(np.uint8))

    fig, axes = plt.subplots(2, 5, figsize=(18.2, 7.8), dpi=230)
    panels = [
        panel_input,
        panel_gt,
        panel_baseline,
        panel_core,
        panel_best,
        panel_final,
        panel_overlay,
        panel_union,
        panel_support,
        panel_diff,
    ]
    titles = [
        "1) RWTD-15 Input",
        "2) GT Partition",
        "3) Baseline Scored",
        "4) Core Mask",
        "5) Best Candidate",
        "6) Final Selected",
        "7) All Candidates Overlay",
        "8) Candidate Overlap Union",
        "9) Proposal Support (Pixel Frequency)",
        "10) Diff Final vs GT",
    ]
    for idx, (ax, panel, title) in enumerate(zip(axes.flatten(), panels, titles)):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.8, fontweight="bold", pad=6)

    fig.subplots_adjust(left=0.006, right=0.998, top=0.94, bottom=0.03, wspace=0.022, hspace=0.14)
    fig.savefig(FIG_DIR / "fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def _safe_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    u = np.logical_or(aa, bb).sum()
    if u == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / u)


def _choose_target_partition(gt_mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    gt_bool = gt_mask.astype(bool)
    ref_bool = ref_mask.astype(bool)
    if _safe_iou(ref_bool, gt_bool) >= _safe_iou(ref_bool, ~gt_bool):
        return gt_bool
    return ~gt_bool


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.30, thickness: int = 2) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = image.copy().astype(np.float32)
    col = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * col
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out_u8, cnts, -1, color, thickness)
    return out_u8


def _load_dense_bank_masks(image_id: int, shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape
    masks: list[np.ndarray] = []
    candidates = sorted(DENSE_BANK_DIR.glob(f"mask_*_{image_id}.png"))
    if not candidates:
        candidates = sorted(BASELINE_DIR.glob(f"mask_*_{image_id}.png"))
    for p in candidates:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(m > 0)
    return masks


def _proposal_feature(image: np.ndarray, gray: np.ndarray, lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() < 16:
        return np.zeros(10, dtype=np.float32)
    pix_lab = lab[m]
    mean = pix_lab.mean(axis=0)
    std = pix_lab.std(axis=0)
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    grad_mean = float(np.abs(grad[m]).mean())
    ys, xs = np.where(m)
    h = max(int(ys.max() - ys.min() + 1), 1)
    w = max(int(xs.max() - xs.min() + 1), 1)
    aspect = w / h
    area = float(m.mean())
    return np.array(
        [
            mean[0],
            mean[1],
            mean[2],
            std[0],
            std[1],
            std[2],
            grad_mean,
            area,
            aspect,
            float(np.sqrt(area)),
        ],
        dtype=np.float32,
    )


def _masked_thumbnail(image: np.ndarray, mask: np.ndarray, out_w: int, out_h: int, frame_color: tuple[int, int, int]) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() < 16:
        thumb = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    else:
        ys, xs = np.where(m)
        y0 = max(int(ys.min()) - 8, 0)
        y1 = min(int(ys.max()) + 9, image.shape[0])
        x0 = max(int(xs.min()) - 8, 0)
        x1 = min(int(xs.max()) + 9, image.shape[1])
        crop = image[y0:y1, x0:x1]
        crop_m = m[y0:y1, x0:x1]
        thumb = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        thumb_m = cv2.resize(crop_m.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
        dim = (thumb.astype(np.float32) * 0.36).astype(np.uint8)
        dim[thumb_m] = thumb[thumb_m]
        thumb = dim
    cv2.rectangle(thumb, (1, 1), (out_w - 2, out_h - 2), frame_color, 2)
    return thumb


def _build_montage_panel(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    labels: list[str] | None = None,
) -> np.ndarray:
    h, w = image.shape[:2]
    panel = np.full((h, w, 3), 246, dtype=np.uint8)
    rows, cols = 2, 3
    gap = 4
    tile_w = (w - (cols + 1) * gap) // cols
    tile_h = (h - (rows + 1) * gap) // rows
    for idx in range(min(6, len(masks))):
        r = idx // cols
        c = idx % cols
        x0 = gap + c * (tile_w + gap)
        y0 = gap + r * (tile_h + gap)
        thumb = _masked_thumbnail(image, masks[idx], tile_w, tile_h, colors[idx % len(colors)])
        if labels is not None and idx < len(labels):
            cv2.putText(
                thumb,
                labels[idx],
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                thumb,
                labels[idx],
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                colors[idx % len(colors)],
                1,
                cv2.LINE_AA,
            )
        panel[y0 : y0 + tile_h, x0 : x0 + tile_w] = thumb
    return panel


def _connected_component_from_adj(adj: np.ndarray, seed: int) -> np.ndarray:
    n = int(adj.shape[0])
    seen = np.zeros(n, dtype=bool)
    stack = [int(seed)]
    seen[int(seed)] = True
    while stack:
        i = int(stack.pop())
        neigh = np.where(adj[i])[0]
        for j in neigh:
            jj = int(j)
            if not seen[jj]:
                seen[jj] = True
                stack.append(jj)
    return seen


def _overlay_masks_with_ids(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    labels: list[str],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    panel = image.copy().astype(np.float32)
    centers: list[tuple[int, int]] = []
    for idx, (mask, color) in enumerate(zip(masks, colors)):
        panel[mask] = 0.74 * panel[mask] + 0.26 * np.array(color, dtype=np.float32)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, cnts, -1, color, 2)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            cx, cy = panel.shape[1] // 2, panel.shape[0] // 2
        else:
            cx, cy = int(xs.mean()), int(ys.mean())
        centers.append((cx, cy))
        cv2.circle(panel, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        if idx < len(labels):
            cv2.putText(
                panel,
                labels[idx],
                (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                labels[idx],
                (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )
    return np.clip(panel, 0, 255).astype(np.uint8), centers


def _build_compatibility_graph_panel(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    sim: np.ndarray,
    labels: list[str],
    threshold: float,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    panel = (0.74 * panel + 0.26 * image).astype(np.uint8)
    n = len(masks)
    centers: list[tuple[int, int]] = []
    for mask, color in zip(masks, colors):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, cnts, -1, color, 1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            centers.append((panel.shape[1] // 2, panel.shape[0] // 2))
        else:
            centers.append((int(xs.mean()), int(ys.mean())))

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= threshold:
                cv2.line(panel, centers[i], centers[j], (248, 248, 248), 1, cv2.LINE_AA)

    for idx, color in enumerate(colors[:n]):
        cx, cy = centers[idx]
        cv2.circle(panel, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        if idx < len(labels):
            cv2.putText(
                panel,
                labels[idx],
                (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                labels[idx],
                (cx + 8, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )
    return panel


def _build_compatibility_panel(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: list[tuple[int, int, int]],
    sim: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    panel = (0.72 * panel + 0.28 * image).astype(np.uint8)
    centers: list[tuple[int, int]] = []
    for mask, color in zip(masks, colors):
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, cnts, -1, color, 1)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            centers.append((panel.shape[1] // 2, panel.shape[0] // 2))
        else:
            centers.append((int(xs.mean()), int(ys.mean())))

    n = len(masks)
    if n >= 2:
        upper = sim[np.triu_indices(n, k=1)]
        if len(upper) > 0:
            thr = max(float(np.quantile(upper, 0.72)), 0.84)
            for i in range(n):
                for j in range(i + 1, n):
                    if float(sim[i, j]) >= thr:
                        cv2.line(panel, centers[i], centers[j], (248, 248, 248), 1, cv2.LINE_AA)

    for idx, color in enumerate(colors[:n]):
        cx, cy = centers[idx]
        cv2.circle(panel, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(panel, (cx, cy), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return panel


def _build_threeway_panel(image: np.ndarray, under: np.ndarray, over: np.ndarray, final: np.ndarray, target: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    under_img = _overlay_boundary(_overlay_mask(image, under, (72, 133, 237), alpha=0.30), target)
    over_img = _overlay_boundary(_overlay_mask(image, over, (219, 68, 55), alpha=0.26), target)
    final_img = _overlay_boundary(_overlay_mask(image, final, (244, 160, 37), alpha=0.32), target)

    sep = np.full((h, 3, 3), 255, dtype=np.uint8)
    tile_w = (w - 6) // 3
    under_t = cv2.resize(under_img, (tile_w, h), interpolation=cv2.INTER_AREA)
    over_t = cv2.resize(over_img, (tile_w, h), interpolation=cv2.INTER_AREA)
    final_t = cv2.resize(final_img, (tile_w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([under_t, sep, over_t, sep, final_t], axis=1)


def _shared_compatibility_story(image_id: int, width: int = 360, height: int = 320, n_select: int = 8) -> dict[str, object]:
    img, gt = _load_rwtd_pair_no_crop(image_id, width=width, height=height)

    core_p = CORE_DIR / f"{image_id}.png"
    core_raw = cv2.imread(str(core_p), cv2.IMREAD_GRAYSCALE)
    if core_raw is None:
        raise FileNotFoundError(core_p)
    core_mask = cv2.resize(core_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    final_p = OURS_DIR / f"mask_0_{image_id}.png"
    final_raw = cv2.imread(str(final_p), cv2.IMREAD_GRAYSCALE)
    if final_raw is None:
        raise FileNotFoundError(final_p)
    final_mask = cv2.resize(final_raw, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    target = _choose_target_partition(gt > 0, final_mask)

    proposals = _load_dense_bank_masks(image_id, img.shape[:2])
    if len(proposals) < n_select:
        raise RuntimeError(f"Not enough proposals for image {image_id} in dense bank")

    scores = np.array([_safe_iou(m, target) for m in proposals], dtype=np.float32)
    areas = np.array([float(m.mean()) for m in proposals], dtype=np.float32)

    min_pool = max(2 * n_select, 12)
    idx_pool = [i for i, (s, a) in enumerate(zip(scores, areas)) if 0.02 <= s <= 0.70 and 0.002 <= a <= 0.70]
    if len(idx_pool) < min_pool:
        idx_pool = [i for i, (s, a) in enumerate(zip(scores, areas)) if s <= 0.80 and 0.001 <= a <= 0.80]
    if len(idx_pool) < min_pool:
        idx_pool = list(range(len(proposals)))
    idx_pool = sorted(idx_pool, key=lambda i: float(scores[i]), reverse=True)[:24]

    selected: list[int] = []
    for i in idx_pool:
        if all(_safe_iou(proposals[i], proposals[j]) < 0.90 for j in selected):
            selected.append(i)
        if len(selected) >= n_select:
            break
    if len(selected) < n_select:
        for i in sorted(range(len(proposals)), key=lambda j: float(scores[j]), reverse=True):
            if i not in selected:
                selected.append(i)
            if len(selected) >= n_select:
                break
    selected = selected[:n_select]

    sel_masks = [proposals[i] for i in selected]
    sel_scores = scores[selected]
    palette = [
        (66, 133, 244),
        (52, 168, 83),
        (244, 180, 0),
        (171, 71, 188),
        (0, 172, 193),
        (245, 124, 0),
        (124, 179, 66),
        (239, 83, 80),
        (41, 121, 255),
        (92, 107, 192),
        (102, 187, 106),
        (255, 167, 38),
    ]
    sel_colors = [palette[i % len(palette)] for i in range(len(sel_masks))]
    labels = [f"P{i+1}" for i in range(len(sel_masks))]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    feats = np.stack([_proposal_feature(img, gray, lab, m) for m in sel_masks], axis=0)
    feats = feats - feats.mean(axis=0, keepdims=True)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim_raw = np.clip(feats @ feats.T, -1.0, 1.0)
    np.fill_diagonal(sim_raw, 1.0)

    seed_idx = int(np.argmax(sel_scores))
    sim_seed = sim_raw[seed_idx]
    compat_thr = max(float(np.quantile(sim_seed, 0.58)), 0.55)
    compat_idx = np.where(sim_seed >= compat_thr)[0]
    if len(compat_idx) < 3:
        compat_idx = np.argsort(sim_seed)[-3:]

    stack = np.stack([sel_masks[int(i)] for i in compat_idx], axis=0).astype(np.uint8)
    vote = stack.sum(axis=0)
    vote_thr = max(2, int(np.ceil(0.60 * len(compat_idx))))
    compat_core = vote >= vote_thr
    if compat_core.sum() < 64:
        compat_core = vote >= max(1, len(compat_idx) // 2)
    compat_union = np.zeros_like(compat_core)
    for i in compat_idx:
        compat_union |= sel_masks[int(i)]

    over_merge = np.zeros_like(compat_core)
    for i in idx_pool[:12]:
        over_merge |= proposals[int(i)]

    sim_vis = 0.5 * (sim_raw + 1.0)
    np.fill_diagonal(sim_vis, 1.0)

    return {
        "img": img,
        "target": target,
        "core_mask": core_mask,
        "final_mask": final_mask,
        "sel_masks": sel_masks,
        "sel_colors": sel_colors,
        "labels": labels,
        "sim_raw": sim_raw,
        "sim_vis": sim_vis,
        "compat_thr": compat_thr,
        "compat_idx": compat_idx,
        "compat_core": compat_core,
        "compat_union": compat_union,
        "over_merge": over_merge,
    }


def build_consolidation_logic_clear_figure() -> None:
    # Two-row figure with non-redundant step-by-step progression.
    image_id = 74
    story = _shared_compatibility_story(image_id=image_id, width=360, height=320, n_select=8)
    img = story["img"]
    target = story["target"]
    sel_masks = story["sel_masks"]
    sel_colors = story["sel_colors"]
    sim_raw = story["sim_raw"]
    sim_vis = story["sim_vis"]
    compat_thr = float(story["compat_thr"])
    compat_core = story["compat_core"]
    compat_union = story["compat_union"]
    compat_idx = np.array(story["compat_idx"], dtype=int)
    over_merge = story["over_merge"]
    final_mask = story["final_mask"]

    # Distinct step panels (no duplicated views).
    panel_input = _overlay_boundary(_overlay_mask(img, target, (56, 190, 88), alpha=0.20, thickness=2), target)
    panel_bank = _overlay_masks(img, sel_masks, colors=sel_colors, alpha=0.24)
    panel_bank = _overlay_boundary(panel_bank, target)
    panel_montage = _build_montage_panel(img, sel_masks, sel_colors)
    panel_graph = _build_compatibility_graph_panel(img, sel_masks, sel_colors, sim_raw, labels=[], threshold=compat_thr)
    panel_union = _overlay_boundary(_overlay_mask(img, compat_union, (245, 194, 66), alpha=0.30, thickness=2), target)
    panel_core = _overlay_boundary(_overlay_mask(img, compat_core, (66, 133, 244), alpha=0.31, thickness=2), target)

    # Pixel-frequency vote inside compatible subset.
    if len(compat_idx) > 0:
        comp_stack = np.stack([sel_masks[int(i)] for i in compat_idx], axis=0).astype(np.float32)
        vote_density = comp_stack.mean(axis=0)
    else:
        vote_density = np.zeros(img.shape[:2], dtype=np.float32)
    heat = np.clip(vote_density * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    panel_vote = (0.65 * img + 0.35 * heat_color).astype(np.uint8)
    panel_vote = _overlay_boundary(panel_vote, target)

    added = np.logical_and(compat_union, ~compat_core)
    panel_added = _overlay_mask(img, compat_core, (66, 133, 244), alpha=0.24, thickness=2)
    panel_added = _overlay_mask(panel_added, added, (245, 158, 11), alpha=0.36, thickness=2)
    panel_added = _overlay_boundary(panel_added, target)

    panel_under = _overlay_boundary(_overlay_mask(img, compat_core, (72, 133, 237), alpha=0.31, thickness=2), target)
    panel_over = _overlay_boundary(_overlay_mask(img, over_merge, (219, 68, 55), alpha=0.28, thickness=2), target)
    panel_final = _overlay_boundary(_overlay_mask(img, final_mask, (244, 140, 35), alpha=0.33, thickness=2), target)

    fig, axes = plt.subplots(2, 6, figsize=(19.8, 8.0), dpi=230)

    # Top row: evidence -> compatibility.
    axes[0, 0].imshow(panel_input)
    axes[0, 1].imshow(panel_bank)
    axes[0, 2].imshow(panel_montage)
    im = axes[0, 3].imshow(sim_vis, cmap="magma", vmin=0.35, vmax=1.0)
    axes[0, 4].imshow(panel_graph)
    axes[0, 5].imshow(panel_union)
    top_titles = [
        "1) RWTD Input",
        "2) Frozen Proposal Bank",
        "3) Masked Proposal Encoding",
        "4) Encoded Pairwise Similarity $s_{ij}$",
        "5) Compatibility Graph",
        "6) Compatible Union Evidence",
    ]

    # Bottom row: consolidation -> decision.
    axes[1, 0].imshow(panel_vote)
    axes[1, 1].imshow(panel_core)
    axes[1, 2].imshow(panel_added)
    axes[1, 3].imshow(panel_under)
    axes[1, 4].imshow(panel_over)
    axes[1, 5].imshow(panel_final)
    bottom_titles = [
        "7) Vote Density in Compatible Set",
        "8) Conservative Core",
        "9) Safe Added Region",
        "10) Under-Merge Baseline",
        "11) Over-Merge Baseline",
        "12) Final Selected Mask",
    ]

    for c in range(6):
        axes[0, c].set_title(top_titles[c], fontsize=9.8, fontweight="bold", pad=6)
        axes[1, c].set_title(bottom_titles[c], fontsize=9.8, fontweight="bold", pad=6)
        if c != 3:
            axes[0, c].set_xticks([])
            axes[0, c].set_yticks([])
        axes[1, c].set_xticks([])
        axes[1, c].set_yticks([])

    n = len(sel_masks)
    axes[0, 3].set_xticks(np.arange(n))
    axes[0, 3].set_yticks(np.arange(n))
    axes[0, 3].set_xticklabels([f"P{i+1}" for i in range(n)], fontsize=7.0)
    axes[0, 3].set_yticklabels([f"P{i+1}" for i in range(n)], fontsize=7.0)
    axes[0, 3].tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes[0, 3], fraction=0.047, pad=0.015)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("compatibility", fontsize=7.5)

    fig.subplots_adjust(left=0.008, right=0.996, top=0.95, bottom=0.03, wspace=0.04, hspace=0.15)
    fig.savefig(FIG_DIR / "fig_consolidation_logic.png", bbox_inches="tight")
    plt.close(fig)


def build_compatibility_reasoning_figure() -> None:
    image_id = 74
    story = _shared_compatibility_story(image_id=image_id, width=360, height=320, n_select=8)
    img = story["img"]
    sel_masks = story["sel_masks"]
    sel_colors = story["sel_colors"]
    labels = story["labels"]
    sim_raw = story["sim_raw"]
    sim_vis = story["sim_vis"]
    compat_thr = float(story["compat_thr"])
    compat_core = story["compat_core"]
    compat_union = story["compat_union"]
    final_mask = story["final_mask"]

    panel_frag, _ = _overlay_masks_with_ids(img, sel_masks, sel_colors, labels)
    panel_montage = _build_montage_panel(img, sel_masks, sel_colors, labels=labels)
    panel_graph = _build_compatibility_graph_panel(img, sel_masks, sel_colors, sim_raw, labels, compat_thr)

    panel_core = _overlay_mask(img, compat_union, (245, 194, 66), alpha=0.20, thickness=2)
    panel_core = _overlay_mask(panel_core, compat_core, (66, 133, 244), alpha=0.32, thickness=2)
    cnts, _ = cv2.findContours(final_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_core, cnts, -1, (244, 140, 35), 2)

    fig, axes = plt.subplots(1, 5, figsize=(16.8, 4.4), dpi=240)
    axes[0].imshow(panel_frag)
    axes[1].imshow(panel_montage)
    im = axes[2].imshow(sim_vis, cmap="magma", vmin=0.35, vmax=1.0)
    axes[3].imshow(panel_graph)
    axes[4].imshow(panel_core)

    titles = [
        "1) Selected Frozen Fragments",
        "2) Masked Proposal Embeddings",
        "3) Pairwise Compatibility $s_{ij}$",
        "4) Thresholded Compatibility Graph",
        "5) Compatible Consensus Core",
    ]
    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx], fontsize=10.3, fontweight="bold", pad=7)
        if idx != 2:
            ax.set_xticks([])
            ax.set_yticks([])

    n = len(sel_masks)
    axes[2].set_xticks(np.arange(n))
    axes[2].set_yticks(np.arange(n))
    axes[2].set_xticklabels(labels, fontsize=7.8)
    axes[2].set_yticklabels(labels, fontsize=7.8)
    axes[2].tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("compatibility", fontsize=8)

    fig.subplots_adjust(left=0.01, right=0.995, top=0.89, bottom=0.06, wspace=0.06)
    fig.savefig(FIG_DIR / "fig_compatibility_reasoning.png", bbox_inches="tight")
    plt.close(fig)


class PTDSynthBuilder:
    def __init__(self, seed: int = 20260307):
        self.rng = np.random.default_rng(seed)
        self.backend = PTDImageBackend(PTD_ROOT)
        _, entries = load_ptd_entries(PTD_ROOT)
        self.by_class = group_entries_by_class(entries)
        self.class_ids = sorted(self.by_class.keys())

    def sample_class(self, forbidden: set[int] | None = None) -> int:
        forbidden = forbidden or set()
        cand = [cid for cid in self.class_ids if cid not in forbidden]
        if not cand:
            raise RuntimeError("No PTD class candidates available")
        return int(cand[self.rng.integers(0, len(cand))])

    @staticmethod
    def _texture_score(patch: np.ndarray) -> float:
        g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_32F)
        return float(np.var(lap))

    def _mosaicize(self, patch: np.ndarray, tile: int = 26) -> np.ndarray:
        h, w = patch.shape[:2]
        out = np.zeros_like(patch)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                th = y1 - y0
                tw = x1 - x0
                sy = int(self.rng.integers(0, max(h - th + 1, 1)))
                sx = int(self.rng.integers(0, max(w - tw + 1, 1)))
                out[y0:y1, x0:x1] = patch[sy : sy + th, sx : sx + tw]
        return out

    def sample_patch(self, class_id: int, width: int, height: int, textureize: bool = False) -> np.ndarray:
        entries = self.by_class[int(class_id)]
        best_patch: np.ndarray | None = None
        best_score = -1.0
        for _ in range(10):
            entry = entries[int(self.rng.integers(0, len(entries)))]
            rgb = self.backend.read_rgb(entry.rel_path)
            h, w = rgb.shape[:2]
            side = min(h, w)
            if side < 24:
                cand = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            else:
                y0 = int(self.rng.integers(0, max(h - side + 1, 1)))
                x0 = int(self.rng.integers(0, max(w - side + 1, 1)))
                crop = rgb[y0 : y0 + side, x0 : x0 + side]
                cand = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
            score = self._texture_score(cand)
            if score > best_score:
                best_score = score
                best_patch = cand
        if best_patch is None:
            entry = entries[0]
            best_patch = cv2.resize(self.backend.read_rgb(entry.rel_path), (width, height), interpolation=cv2.INTER_AREA)
        if textureize:
            best_patch = self._mosaicize(best_patch)
        return best_patch

    def disconnected_scene(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        bg_cls = self.sample_class()
        tgt_cls = self.sample_class({bg_cls})
        bg = self.sample_patch(bg_cls, width, height, textureize=True)
        tgt = self.sample_patch(tgt_cls, width, height, textureize=True)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (int(width * 0.30), int(height * 0.47)), (int(width * 0.16), int(height * 0.22)), 12, 0, 360, 1, -1)
        cv2.ellipse(mask, (int(width * 0.72), int(height * 0.54)), (int(width * 0.15), int(height * 0.20)), -18, 0, 360, 1, -1)
        scene = bg.copy()
        scene[mask > 0] = tgt[mask > 0]
        return scene, mask > 0

    def weak_boundary_scene(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        left_cls = self.sample_class()
        right_cls = self.sample_class({left_cls})
        left = self.sample_patch(left_cls, width, height, textureize=True).astype(np.float32)
        right = self.sample_patch(right_cls, width, height, textureize=True).astype(np.float32)

        ys = np.arange(height, dtype=np.float32)
        boundary = width * 0.50 + 22.0 * np.sin(ys / 23.0)
        x = np.arange(width, dtype=np.float32)[None, :]
        b = boundary[:, None]
        dist = b - x
        alpha = np.clip((dist + 20.0) / 40.0, 0.0, 1.0)  # soft transition zone
        scene = (alpha[..., None] * left + (1.0 - alpha[..., None]) * right).astype(np.uint8)
        mask = dist > 0
        return scene, mask

    def repeated_pattern_scene(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        tgt_cls = self.sample_class()
        oth_cls = self.sample_class({tgt_cls})
        tgt = self.sample_patch(tgt_cls, width, height, textureize=True)
        oth = self.sample_patch(oth_cls, width, height, textureize=True)
        scene = oth.copy()
        mask = np.zeros((height, width), dtype=np.uint8)

        # Main target partition: broad connected region with a curved boundary.
        poly = np.array(
            [
                [int(width * 0.10), int(height * 0.20)],
                [int(width * 0.44), int(height * 0.12)],
                [int(width * 0.82), int(height * 0.24)],
                [int(width * 0.78), int(height * 0.64)],
                [int(width * 0.52), int(height * 0.88)],
                [int(width * 0.18), int(height * 0.78)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [poly], 1)
        cv2.ellipse(mask, (int(width * 0.56), int(height * 0.48)), (int(width * 0.20), int(height * 0.16)), -24, 0, 360, 1, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
        mask_bool = mask > 0
        scene[mask_bool] = tgt[mask_bool]

        # Add look-alike distractors outside the target partition.
        for cx, cy, rx, ry, ang in [
            (0.14, 0.84, 0.10, 0.07, 18),
            (0.87, 0.20, 0.09, 0.08, -14),
            (0.84, 0.79, 0.11, 0.07, 27),
            (0.16, 0.17, 0.08, 0.07, -8),
        ]:
            blob = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                blob,
                (int(width * cx), int(height * cy)),
                (max(4, int(width * rx)), max(4, int(height * ry))),
                ang,
                0,
                360,
                1,
                -1,
            )
            distract = np.logical_and(blob > 0, ~mask_bool)
            scene[distract] = (0.82 * tgt[distract] + 0.18 * oth[distract]).astype(np.uint8)

        return scene, mask_bool


def _overlay_target(scene: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (240, 140, 35)) -> np.ndarray:
    out = scene.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out[mask] = 0.70 * out[mask] + 0.30 * c
    out = out.astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 3)
    return out


def build_rwtd_factors_figure() -> None:
    specs = [
        FactorSpec(71, "A. Weak Boundaries", "Boundary cues are subtle and non-semantic."),
        FactorSpec(469, "B. Disconnected Regions", "One texture appears in separated image regions."),
        FactorSpec(270, "C. Repeated Patterns", "Local repeats encourage unsafe unions."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.8), dpi=220)
    for ax, spec in zip(axes, specs):
        panel = _load_rwtd_overlay(spec.image_id)
        ax.imshow(panel)
        ax.set_title(f"{spec.title}\nRWTD ID {spec.image_id}", fontsize=12, pad=8, fontweight="bold")
        _draw_text(ax, spec.note)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Three real-world factors that make texture segmentation hard on RWTD",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.84, bottom=0.03, wspace=0.02)
    fig.savefig(FIG_DIR / "fig_rwtd_three_factors.png", bbox_inches="tight")
    plt.close(fig)


def build_ptd_bridge_figure() -> None:
    builder = PTDSynthBuilder()
    rwtd_specs = [
        FactorSpec(71, "Weak boundary", "Soft boundary transitions"),
        FactorSpec(469, "Disconnected target", "Same texture in multiple components"),
        FactorSpec(270, "Repeated pattern", "Look-alike distractor regions"),
    ]
    synth_builders = [
        builder.weak_boundary_scene,
        builder.disconnected_scene,
        builder.repeated_pattern_scene,
    ]

    fig = plt.figure(figsize=(23.8, 4.5), dpi=220)
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.11, 1, 1, 1])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 4]),
        fig.add_subplot(gs[0, 5]),
        fig.add_subplot(gs[0, 6]),
    ]

    panels: list[tuple[np.ndarray, str, str]] = []
    for spec in rwtd_specs:
        panels.append(
            (
                _load_rwtd_overlay(spec.image_id, width=520, height=320),
                f"RWTD: {spec.title}",
                f"ID {spec.image_id}",
            )
        )
    for spec, synth_fn in zip(rwtd_specs, synth_builders):
        bot_scene, bot_mask = synth_fn(520, 320)
        panels.append(
            (
                _overlay_target(bot_scene, bot_mask),
                f"PTD: {spec.title}",
                spec.note,
            )
        )

    for ax, (panel, title, note) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.4, fontweight="bold", pad=7)
        _draw_text(ax, note)

    fig.text(0.255, 0.98, "RWTD Failure Factors", ha="center", va="top", fontsize=14.5, fontweight="bold")
    fig.text(0.758, 0.98, "Matched PTD Synthetic Counterparts", ha="center", va="top", fontsize=14.5, fontweight="bold")
    fig.subplots_adjust(left=0.012, right=0.995, top=0.88, bottom=0.03, wspace=0.018)
    fig.savefig(FIG_DIR / "fig_ptd_rwtd_bridge.png", bbox_inches="tight")
    plt.close(fig)


def build_ambiguity_commitment_figure() -> None:
    if not FAILURE_AUDIT_CSV.exists() or not PER_IMAGE_CSV.exists():
        raise FileNotFoundError("Missing full-256 audit CSVs for ambiguity-commitment figure.")

    fa = pd.read_csv(FAILURE_AUDIT_CSV)
    pi = pd.read_csv(PER_IMAGE_CSV, usecols=["image_id", "acute_iou", "acute_ari"])
    df = fa.merge(pi, on="image_id", how="left")
    df["margin_abs"] = df["decision_margin"].abs()

    # Panel A: wrong-partition rate by absolute decision-margin bins.
    bins = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.00], dtype=np.float32)
    labels = ["0-0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40-1.00"]
    df["margin_bin"] = pd.cut(df["margin_abs"], bins=bins, include_lowest=True, labels=labels)
    grp = (
        df.groupby("margin_bin", observed=False)
        .agg(
            n=("image_id", "count"),
            wrong_rate=("wrong_partition_commitment_flag", "mean"),
        )
        .reset_index()
    )
    wrong_rate_pct = 100.0 * grp["wrong_rate"].to_numpy(dtype=np.float32)
    counts = grp["n"].to_numpy(dtype=np.int32)

    # Panel B: cumulative wrong-partition rate as low-margin cases are included.
    s = df.sort_values("margin_abs", ascending=False).reset_index(drop=True)
    coverage = 100.0 * (np.arange(len(s), dtype=np.float32) + 1.0) / float(len(s))
    cum_wrong = s["wrong_partition_commitment_flag"].to_numpy(dtype=np.float32).cumsum()
    cum_wrong_rate = 100.0 * cum_wrong / (np.arange(len(s), dtype=np.float32) + 1.0)

    # Panel C: top-k recoverability summary.
    all_n = len(df)
    all_k = int(df["topk_oracle_recoverable"].sum())
    low_iou = df[df["acute_iou"] < 0.5]
    low_n = len(low_iou)
    low_k = int(low_iou["topk_oracle_recoverable"].sum()) if low_n > 0 else 0
    wrong = df[df["wrong_partition_commitment_flag"] == 1]
    wrong_n = len(wrong)
    wrong_k = int(wrong["topk_oracle_recoverable"].sum()) if wrong_n > 0 else 0

    rec_labels = ["All images", "Low-IoU cases", "Wrong-partition cases"]
    rec_den = np.array([all_n, max(low_n, 1), max(wrong_n, 1)], dtype=np.float32)
    rec_num = np.array([all_k, low_k, wrong_k], dtype=np.float32)
    rec_pct = 100.0 * rec_num / rec_den
    rec_ann = [f"{all_k}/{all_n}", f"{low_k}/{low_n}", f"{wrong_k}/{wrong_n}"]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.7), dpi=230)

    # A
    x = np.arange(len(labels))
    axes[0].bar(x, wrong_rate_pct, color="#d66b5d", edgecolor="black", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0, fontsize=8)
    axes[0].set_ylabel("Wrong-partition rate (%)", fontsize=9)
    axes[0].set_xlabel(r"$|$decision margin$|$ bin", fontsize=9)
    axes[0].set_title("A) Wrong Commitment vs Margin", fontsize=10, pad=6)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.7)
    for i, (v, n) in enumerate(zip(wrong_rate_pct, counts)):
        axes[0].text(i, v + 0.7, f"n={n}", ha="center", va="bottom", fontsize=7)

    # B
    axes[1].plot(coverage, cum_wrong_rate, color="#2d5f9a", linewidth=2.0)
    axes[1].set_xlabel("Retained coverage (%)", fontsize=9)
    axes[1].set_ylabel("Cumulative wrong-partition rate (%)", fontsize=9)
    axes[1].set_title("B) Hard-Commitment Risk vs Coverage", fontsize=10, pad=6)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, max(2.0, float(cum_wrong_rate.max()) + 0.8))
    axes[1].grid(alpha=0.22, linewidth=0.7)
    first_wrong = np.where(cum_wrong > 0)[0]
    if len(first_wrong) > 0:
        fw = int(first_wrong[0])
        axes[1].scatter([coverage[fw]], [cum_wrong_rate[fw]], color="#d66b5d", s=22, zorder=3)
        axes[1].text(
            float(coverage[fw]) + 1.2,
            float(cum_wrong_rate[fw]) + 0.08,
            "first wrong commitment",
            fontsize=7.2,
            color="#7a352b",
        )

    # C
    xr = np.arange(len(rec_labels))
    axes[2].bar(xr, rec_pct, color=["#4c78a8", "#72b7b2", "#54a24b"], edgecolor="black", linewidth=0.7)
    axes[2].set_xticks(xr)
    axes[2].set_xticklabels(rec_labels, rotation=0, fontsize=8)
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("Top-k recoverable (%)", fontsize=9)
    axes[2].set_title("C) Proposal-Bank Recoverability", fontsize=10, pad=6)
    axes[2].grid(axis="y", alpha=0.22, linewidth=0.7)
    for i, (v, ann) in enumerate(zip(rec_pct, rec_ann)):
        axes[2].text(i, min(108.0, v + 2.0), ann, ha="center", va="bottom", fontsize=7.2)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.18, wspace=0.32)
    fig.savefig(FIG_DIR / "fig_ambiguity_commitment_full256.png", bbox_inches="tight")
    plt.close(fig)


def _extract_two_row_audit_bands(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    bg = rgb[0, 0].astype(np.int16)
    diff = np.abs(rgb.astype(np.int16) - bg).sum(axis=2)
    active = diff.mean(axis=1) > 2.0

    bands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start > 10:
                bands.append((start, idx - 1))
            start = None
    if start is not None:
        bands.append((start, len(active) - 1))

    if len(bands) < 4:
        raise RuntimeError(f"Could not detect two-row content bands in {path}")

    top = rgb[bands[0][0] : bands[1][1] + 1]
    bottom = rgb[bands[2][0] : bands[3][1] + 1]
    return top, bottom


def build_route_audits_figure() -> None:
    rwtd_top, rwtd_bottom = _extract_two_row_audit_bands(FIG_DIR / "fig_pipeline.png")
    detexture_top, detexture_bottom = _extract_two_row_audit_bands(FIG_DIR / "fig_detexture_audit.png")
    stld_top, stld_bottom = _extract_two_row_audit_bands(FIG_DIR / "fig_stld_audit.png")
    controlnet_top, controlnet_bottom = _extract_two_row_audit_bands(FIG_DIR / "fig_controlnet_audit.png")

    gap_small = 16
    gap_group = 34
    bg = np.array([217, 217, 217], dtype=np.uint8)
    route_rows = [
        (rwtd_top, rwtd_bottom),
        (detexture_top, detexture_bottom),
        (stld_top, stld_bottom),
        (controlnet_top, controlnet_bottom),
    ]
    width = max(block.shape[1] for pair in route_rows for block in pair)
    height = 0
    for idx, (top, bottom) in enumerate(route_rows):
        height += top.shape[0] + gap_small + bottom.shape[0]
        if idx + 1 < len(route_rows):
            height += gap_group
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)

    y = 0
    for idx, (top, bottom) in enumerate(route_rows):
        for block, gap in [
            (top, gap_small),
            (bottom, gap_group if idx + 1 < len(route_rows) else 0),
        ]:
            h, w = block.shape[:2]
            x = max((width - w) // 2, 0)
            canvas[y : y + h, x : x + w] = block
            y += h + gap

    fig_width = 18.4
    fig_height = 20.2 * (len(route_rows) / 4.0)
    plt.figure(figsize=(fig_width, fig_height), dpi=230)
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(FIG_DIR / "fig_route_audits.png", bbox_inches="tight", pad_inches=0.01)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_rwtd_factors_figure()
    build_ptd_bridge_figure()
    build_pipeline_clear_figure()
    build_consolidation_logic_clear_figure()
    build_compatibility_reasoning_figure()
    build_ambiguity_commitment_figure()
    build_route_audits_figure()
    print(f"wrote {FIG_DIR / 'fig_rwtd_three_factors.png'}")
    print(f"wrote {FIG_DIR / 'fig_ptd_rwtd_bridge.png'}")
    print(f"wrote {FIG_DIR / 'fig_pipeline.png'}")
    print(f"wrote {FIG_DIR / 'fig_consolidation_logic.png'}")
    print(f"wrote {FIG_DIR / 'fig_compatibility_reasoning.png'}")
    print(f"wrote {FIG_DIR / 'fig_ambiguity_commitment_full256.png'}")
    print(f"wrote {FIG_DIR / 'fig_route_audits.png'}")


if __name__ == "__main__":
    main()
