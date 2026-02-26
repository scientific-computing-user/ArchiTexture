#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def _balanced_pick(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows

    random.seed(seed)
    groups: dict[str, list[dict[str, Any]]] = {"selected": [], "borderline": [], "rejected": []}
    for r in rows:
        st = str(r.get("status") or "rejected")
        if st not in groups:
            st = "rejected"
        groups[st].append(r)

    target = {
        "selected": max(1, int(round(max_samples * 0.40))),
        "borderline": max(1, int(round(max_samples * 0.20))),
        "rejected": max(1, int(round(max_samples * 0.40))),
    }

    picked: list[dict[str, Any]] = []
    for k in ("selected", "borderline", "rejected"):
        g = groups[k]
        if not g:
            continue
        g_sorted = sorted(g, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        n = min(target[k], len(g_sorted))
        if k == "rejected" and len(g_sorted) > n:
            half = max(1, n // 2)
            top = g_sorted[:half]
            tail_pool = g_sorted[half:]
            tail_n = n - len(top)
            tail = random.sample(tail_pool, k=min(tail_n, len(tail_pool))) if tail_n > 0 else []
            picked.extend(top + tail)
        else:
            picked.extend(g_sorted[:n])

    if len(picked) < max_samples:
        picked_ids = {str(x.get("image_id")) for x in picked}
        pool = [r for r in rows if str(r.get("image_id")) not in picked_ids]
        pool = sorted(pool, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        picked.extend(pool[: max_samples - len(picked)])

    if len(picked) > max_samples:
        picked = sorted(picked, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)[:max_samples]

    return picked


def _safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_bundle(source_review_dir: Path, out_review_dir: Path, max_samples: int, seed: int) -> dict[str, Any]:
    data_path = source_review_dir / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.json: {data_path}")

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise RuntimeError("data.json is not a list")

    if int(max_samples) <= 0:
        selected = list(rows)
    else:
        selected = _balanced_pick(rows, max_samples=max_samples, seed=seed)

    out_review_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("thumbnails", "masks", "overlays", "originals"):
        (out_review_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy viewer template.
    src_index = source_review_dir / "index.html"
    if not src_index.exists():
        raise FileNotFoundError(f"Missing index.html: {src_index}")
    shutil.copy2(src_index, out_review_dir / "index.html")

    out_rows: list[dict[str, Any]] = []
    copied = {"thumb": 0, "mask": 0, "overlay": 0, "original": 0}

    for r in selected:
        row = dict(r)
        image_id = str(row.get("image_id") or "unknown")
        thumb_out_rel = ""

        # Thumbnail (relative path from source review dir).
        thumb_rel = str(row.get("thumb") or "").strip()
        if thumb_rel:
            src_thumb = source_review_dir / thumb_rel
            thumb_name = f"{image_id}.jpg"
            dst_thumb = out_review_dir / "thumbnails" / thumb_name
            if _safe_copy(src_thumb, dst_thumb):
                thumb_out_rel = f"thumbnails/{thumb_name}"
                row["thumb"] = thumb_out_rel
                copied["thumb"] += 1

        # Mask and overlay are already relative in current review site.
        mask_rel = str(row.get("mask_rel") or "").strip()
        mask_ok = False
        if mask_rel:
            src_mask = source_review_dir / mask_rel
            mask_name = f"{image_id}.jpg"
            dst_mask = out_review_dir / "masks" / mask_name
            if _safe_copy(src_mask, dst_mask):
                row["mask_rel"] = f"masks/{mask_name}"
                copied["mask"] += 1
                mask_ok = True
        if not mask_ok:
            row["mask_rel"] = thumb_out_rel

        overlay_rel = str(row.get("overlay_rel") or "").strip()
        overlay_ok = False
        if overlay_rel:
            src_overlay = source_review_dir / overlay_rel
            overlay_name = f"{image_id}.jpg"
            dst_overlay = out_review_dir / "overlays" / overlay_name
            if _safe_copy(src_overlay, dst_overlay):
                row["overlay_rel"] = f"overlays/{overlay_name}"
                copied["overlay"] += 1
                overlay_ok = True
        if not overlay_ok:
            row["overlay_rel"] = thumb_out_rel

        # Original image path can be absolute; copy into bundle for GitHub Pages.
        img_path = Path(str(row.get("image_path") or "")).expanduser()
        if img_path.exists() and img_path.is_file():
            ext = img_path.suffix.lower() or ".jpg"
            dst_orig = out_review_dir / "originals" / f"{image_id}{ext}"
            if _safe_copy(img_path, dst_orig):
                row["image_path"] = f"originals/{image_id}{ext}"
                copied["original"] += 1
        else:
            # Fallback to thumbnail if original is unavailable.
            row["image_path"] = thumb_out_rel

        out_rows.append(row)

    # Write JSON + JS payload for viewer.
    data_json_path = out_review_dir / "data.json"
    data_js_path = out_review_dir / "data.js"
    data_json_path.write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
    data_js_path.write_text("window.REVIEW_DATA = " + json.dumps(out_rows, ensure_ascii=False) + ";\n", encoding="utf-8")

    by_status: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    for r in out_rows:
        st = str(r.get("status") or "")
        ds = str(r.get("dataset") or "")
        by_status[st] = by_status.get(st, 0) + 1
        by_dataset[ds] = by_dataset.get(ds, 0) + 1

    summary = {
        "source": str(source_review_dir),
        "out": str(out_review_dir),
        "n_input": len(rows),
        "n_output": len(out_rows),
        "copied": copied,
        "by_status": by_status,
        "by_dataset": by_dataset,
    }
    (out_review_dir / "bundle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build compact GitHub Pages bundle from local review site")
    ap.add_argument("--source_review", required=True)
    ap.add_argument("--out_review", required=True)
    ap.add_argument("--max_samples", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    summary = build_bundle(
        source_review_dir=Path(args.source_review),
        out_review_dir=Path(args.out_review),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
