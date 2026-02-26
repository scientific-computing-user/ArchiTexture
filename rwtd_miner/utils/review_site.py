from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

from rwtd_miner.utils.io import ensure_dir


def _make_thumb(src: Path, dst: Path, size: tuple[int, int] = (300, 220)) -> None:
    try:
        with Image.open(src) as im:
            tile = ImageOps.fit(im.convert("RGB"), size, method=Image.Resampling.BICUBIC)
        ensure_dir(dst.parent)
        tile.save(dst, quality=88)
    except Exception:
        pass


def _localize_asset(src_value: Any, fallback_name: str, dst_dir: Path, rel_prefix: str) -> str | None:
    if src_value is None or (isinstance(src_value, float) and pd.isna(src_value)):
        return None
    s = str(src_value).strip()
    if not s:
        return None

    src_path = Path(s)
    if not src_path.is_absolute():
        # Keep already-local relative refs.
        return s

    if not src_path.exists():
        return None

    suffix = src_path.suffix.lower() or ".jpg"
    dst_path = dst_dir / f"{fallback_name}{suffix}"
    if src_path.resolve() == dst_path.resolve():
        return f"{rel_prefix}/{dst_path.name}"
    try:
        ensure_dir(dst_path.parent)
        if dst_path.exists():
            dst_path.unlink()
        shutil.copy2(src_path, dst_path)
        return f"{rel_prefix}/{dst_path.name}"
    except Exception:
        return None


def build_review_site(df: pd.DataFrame, batch_dir: Path) -> Path:
    review_dir = ensure_dir(batch_dir / "review")
    thumbs_dir = ensure_dir(review_dir / "thumbnails")
    masks_dir = ensure_dir(review_dir / "masks")
    overlays_dir = ensure_dir(review_dir / "overlays")

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        image_path = Path(str(r.get("image_path", "")))
        if not image_path.exists():
            continue

        image_id = str(r.get("image_id", ""))
        thumb_rel = f"thumbnails/{image_id}.jpg"
        thumb_abs = thumbs_dir / f"{image_id}.jpg"
        _make_thumb(image_path, thumb_abs)
        mask_rel = _localize_asset(r.get("geom_mask_rel"), image_id, masks_dir, "masks")
        overlay_rel = _localize_asset(r.get("geom_overlay_rel"), image_id, overlays_dir, "overlays")

        stage_d = r.get("stageD_score_0_100")
        stage_b = r.get("stageB_clip_score")
        stage_a = r.get("stageA_rwtd_score")
        review_score = r.get("review_score")

        if pd.notna(review_score):
            final_score = float(review_score)
        elif pd.notna(stage_d):
            final_score = float(stage_d)
        elif pd.notna(stage_b):
            final_score = float(stage_b) * 100.0
        elif pd.notna(stage_a):
            final_score = float(stage_a)
        else:
            final_score = 0.0

        status = "rejected"
        if bool(r.get("final_selected", False)):
            status = "selected"
        elif bool(r.get("final_borderline", False)):
            status = "borderline"

        rows.append(
            {
                "image_id": image_id,
                "dataset": "" if pd.isna(r.get("dataset")) else str(r.get("dataset")),
                "image_path": str(image_path),
                "thumb": thumb_rel,
                "status": status,
                "final_score": round(final_score, 3),
                "selection_reason": "" if pd.isna(r.get("selection_reason")) else str(r.get("selection_reason")),
                "stageA_rwtd_score": None if pd.isna(stage_a) else round(float(stage_a), 3),
                "stageB_clip_score": None if pd.isna(stage_b) else round(float(stage_b), 6),
                "stageD_score_0_100": None if pd.isna(stage_d) else round(float(stage_d), 3),
                "stageA_n_masks": None if pd.isna(r.get("stageA_n_masks")) else int(r.get("stageA_n_masks")),
                "stageA_largest_ratio": None if pd.isna(r.get("stageA_largest_ratio")) else round(float(r.get("stageA_largest_ratio")), 6),
                "stageA_small_frac": None if pd.isna(r.get("stageA_small_frac")) else round(float(r.get("stageA_small_frac")), 6),
                "stageA_pass": bool(r.get("stageA_pass", False)),
                "stageB_pass": bool(r.get("stageB_pass", False)),
                "stageC_pass": None if pd.isna(r.get("stageC_pass")) else bool(r.get("stageC_pass")),
                "stageD_decision": None if pd.isna(r.get("stageD_decision")) else str(r.get("stageD_decision")),
                "geom_texture_boundary_score": None
                if pd.isna(r.get("geom_texture_boundary_score"))
                else round(float(r.get("geom_texture_boundary_score")), 3),
                "geom_object_fraction": None
                if pd.isna(r.get("geom_object_fraction"))
                else round(float(r.get("geom_object_fraction")), 6),
                "geom_texture_fraction": None
                if pd.isna(r.get("geom_texture_fraction"))
                else round(float(r.get("geom_texture_fraction")), 6),
                "geom_num_large_texture_regions": None
                if pd.isna(r.get("geom_num_large_texture_regions"))
                else int(r.get("geom_num_large_texture_regions")),
                "geom_num_strong_boundaries": None
                if pd.isna(r.get("geom_num_strong_boundaries"))
                else int(r.get("geom_num_strong_boundaries")),
                "geom_boundary_norm": None if pd.isna(r.get("geom_boundary_norm")) else round(float(r.get("geom_boundary_norm")), 6),
                "mask_rel": mask_rel,
                "overlay_rel": overlay_rel,
            }
        )

    rows.sort(key=lambda x: x["final_score"], reverse=True)

    (review_dir / "data.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (review_dir / "data.js").write_text(f"window.REVIEW_DATA = {json.dumps(rows, ensure_ascii=False)};\n", encoding="utf-8")

    html = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"Cache-Control\" content=\"no-cache, no-store, must-revalidate\" />
  <meta http-equiv=\"Pragma\" content=\"no-cache\" />
  <meta http-equiv=\"Expires\" content=\"0\" />
  <title>RWTD Miner Review</title>
  <style>
    :root{--bg:#eef2f6;--panel:#fff;--ink:#152235;--line:#d6dee7;--muted:#5b6a7e}
    *{box-sizing:border-box}
    body{margin:0;background:linear-gradient(135deg,#e8eef4,#f6f9fc);font-family:\"Avenir Next\",\"Helvetica Neue\",sans-serif;color:var(--ink)}
    .wrap{max-width:1450px;margin:0 auto;padding:16px}
    .head{display:flex;justify-content:space-between;align-items:center;gap:10px}
    .help{margin:10px 0;background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 12px;color:var(--muted);font-size:13px}
    .toolbar{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:10px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;position:sticky;top:8px;z-index:5}
    label{font-size:12px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
    input,select{height:34px;border:1px solid var(--line);border-radius:8px;padding:4px 8px;background:#fff}
    .meta{display:flex;justify-content:space-between;align-items:center;margin:10px 2px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
    .card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;cursor:pointer}
    .card img{display:block;width:100%;height:158px;object-fit:cover;background:#eef2f6}
    .txt{padding:8px 10px}
    .t{font-weight:700;font-size:13px}
    .s{font-size:12px;color:var(--muted);margin-top:2px}
    .pill{display:inline-block;font-size:11px;border-radius:999px;border:1px solid;padding:2px 8px}
    .selected{color:#0b7b46;border-color:#0b7b4638;background:#0b7b4610}
    .borderline{color:#946100;border-color:#94610038;background:#94610010}
    .rejected{color:#9f1239;border-color:#9f123938;background:#9f123910}
    .detail{position:fixed;right:0;top:0;height:100vh;width:min(980px,100vw);background:#fff;border-left:1px solid var(--line);transform:translateX(100%);transition:.2s;overflow:auto;z-index:20}
    .detail.open{transform:translateX(0)}
    .inner{padding:14px}
    .hero{width:100%;max-height:240px;object-fit:contain;background:#f5f7fa;border:1px solid var(--line);border-radius:10px}
    .imgrow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
    .kv{margin-top:8px;font-size:13px}
    .kv b{display:inline-block;width:210px;color:var(--muted)}
    .btn{height:34px;border:1px solid var(--line);background:#fff;border-radius:8px;padding:0 10px;cursor:pointer}
    @media (max-width:900px){.toolbar{grid-template-columns:repeat(2,minmax(120px,1fr))}.imgrow{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class=\"wrap\">
  <div class=\"head\"><h2>RWTD Miner Review</h2><button class=\"btn\" id=\"reset\">Reset</button></div>
  <div class=\"help\">How to use: set <b>Dataset</b> and <b>Status</b>, keep sort on <b>final score</b>, then open cards to inspect <b>original / mask / boundary overlay</b> and the exact metrics behind the rank.</div>
  <div class=\"toolbar\">
    <label>Dataset<select id=\"f_dataset\"><option value=\"\">All</option></select></label>
    <label>Status<select id=\"f_status\"><option value=\"\">All</option><option value=\"selected\">selected</option><option value=\"borderline\">borderline</option><option value=\"rejected\">rejected</option></select></label>
    <label>Sort<select id=\"f_sort\"><option value=\"final\">final score</option><option value=\"geom\">boundary score</option><option value=\"stageA\">stageA score</option><option value=\"stageB\">stageB score</option><option value=\"masks\">#masks</option></select></label>
    <label>Score min<input type=\"number\" id=\"f_min\" min=\"0\" max=\"100\" value=\"0\"/></label>
    <label>Score max<input type=\"number\" id=\"f_max\" min=\"0\" max=\"100\" value=\"100\"/></label>
    <label>Min masks<input type=\"number\" id=\"f_masks\" min=\"0\" value=\"0\"/></label>
    <label>Object max<input type=\"number\" id=\"f_obj\" min=\"0\" max=\"1\" step=\"0.01\" value=\"1\"/></label>
    <label>ID contains<input type=\"text\" id=\"f_id\" placeholder=\"sa_123\"/></label>
  </div>
  <div class=\"meta\"><div id=\"count\"></div><div>Keyboard: ← prev, → next, Esc close</div></div>
  <div id=\"grid\" class=\"grid\"></div>
</div>
<aside id=\"detail\" class=\"detail\"><div class=\"inner\"></div></aside>
<script src=\"data.js\"></script>
<script>
const state={rows:Array.isArray(window.REVIEW_DATA)?window.REVIEW_DATA:[],filtered:[],idx:-1};
const el={dataset:document.getElementById('f_dataset'),status:document.getElementById('f_status'),sort:document.getElementById('f_sort'),min:document.getElementById('f_min'),max:document.getElementById('f_max'),masks:document.getElementById('f_masks'),obj:document.getElementById('f_obj'),id:document.getElementById('f_id'),grid:document.getElementById('grid'),count:document.getElementById('count'),detail:document.getElementById('detail'),inner:document.querySelector('#detail .inner'),reset:document.getElementById('reset')};
const num=(v,d=0)=>{const n=Number(v);return Number.isFinite(n)?n:d};
function initDatasets(){const set=new Set(state.rows.map(r=>String(r.dataset||'').trim()).filter(Boolean));const vals=[...set].sort();for(const d of vals){const o=document.createElement('option');o.value=d;o.textContent=d;el.dataset.appendChild(o);}}
function sortRows(rows){const s=el.sort.value;const a=[...rows];
 if(s==='final')a.sort((x,y)=>num(y.final_score)-num(x.final_score));
 if(s==='geom')a.sort((x,y)=>num(y.geom_texture_boundary_score)-num(x.geom_texture_boundary_score));
 if(s==='stageA')a.sort((x,y)=>num(y.stageA_rwtd_score)-num(x.stageA_rwtd_score));
 if(s==='stageB')a.sort((x,y)=>num(y.stageB_clip_score)-num(x.stageB_clip_score));
 if(s==='masks')a.sort((x,y)=>num(y.stageA_n_masks)-num(x.stageA_n_masks));
 return a;}
function filterRows(){const ds=el.dataset.value;const st=el.status.value;const smin=num(el.min.value,0);const smax=num(el.max.value,100);const mm=num(el.masks.value,0);const om=num(el.obj.value,1);const id=(el.id.value||'').toLowerCase();
 let rows=state.rows.filter(r=>{if(ds&&String(r.dataset)!==ds)return false;if(st&&r.status!==st)return false;const fs=num(r.final_score);if(fs<smin||fs>smax)return false;if(num(r.stageA_n_masks)<mm)return false;if(num(r.geom_object_fraction,0)>om)return false;if(id&&!String(r.image_id).toLowerCase().includes(id))return false;return true;});
 state.filtered=sortRows(rows);render();}
function pill(s){return `<span class="pill ${s}">${s}</span>`;}
function card(r,i){const ds=r.dataset?`${r.dataset} / `:'';return `<article class=\"card\" data-i=\"${i}\"><img src=\"${r.thumb}\" loading=\"lazy\" onerror=\"this.style.background='#f1f3f7'\"/><div class=\"txt\"><div class=\"t\">${ds}${r.image_id}</div><div class=\"s\">final ${num(r.final_score).toFixed(1)} | boundary ${num(r.geom_texture_boundary_score).toFixed(1)} | obj ${num(r.geom_object_fraction).toFixed(3)}</div><div class=\"s\">${pill(r.status)}</div></div></article>`;}
function render(){el.count.textContent=`${state.filtered.length} samples`;if(!state.filtered.length){el.grid.innerHTML='<div style="grid-column:1/-1;background:#fff;border:1px solid #d6dee7;border-radius:10px;padding:14px;color:#5b6a7e">No samples match current filters.</div>';return;}el.grid.innerHTML=state.filtered.map((r,i)=>card(r,i)).join('');el.grid.querySelectorAll('.card').forEach(c=>c.onclick=()=>openDetail(Number(c.dataset.i)));}
function openDetail(i){if(i<0||i>=state.filtered.length)return;state.idx=i;const r=state.filtered[i];const m=r.mask_rel?r.mask_rel:'';const o=r.overlay_rel?r.overlay_rel:'';el.inner.innerHTML=`<div style="position:sticky;top:0;background:#fff;padding:6px 0 10px"><button class=\"btn\" id=\"c\">Close</button> <button class=\"btn\" id=\"p\">Prev</button> <button class=\"btn\" id=\"n\">Next</button></div><h3>${r.dataset?`${r.dataset} / `:''}${r.image_id}</h3><div class=\"imgrow\"><figure><img class=\"hero\" src=\"${r.image_path}\" onerror=\"this.src='${r.thumb}'\"/><figcaption>Original</figcaption></figure><figure><img class=\"hero\" src=\"${m}\" onerror=\"this.style.background='#f1f3f7'\"/><figcaption>Annotation mask visualization</figcaption></figure><figure><img class=\"hero\" src=\"${o}\" onerror=\"this.style.background='#f1f3f7'\"/><figcaption>Texture-boundary overlay</figcaption></figure></div><div class=\"kv\"><b>dataset</b>${r.dataset||''}</div><div class=\"kv\"><b>status</b>${r.status}</div><div class=\"kv\"><b>selection_reason</b>${r.selection_reason||''}</div><div class=\"kv\"><b>final_score</b>${num(r.final_score).toFixed(2)}</div><div class=\"kv\"><b>stageA_rwtd_score</b>${r.stageA_rwtd_score??''}</div><div class=\"kv\"><b>geom_texture_boundary_score</b>${r.geom_texture_boundary_score??''}</div><div class=\"kv\"><b>geom_object_fraction</b>${r.geom_object_fraction??''}</div><div class=\"kv\"><b>geom_texture_fraction</b>${r.geom_texture_fraction??''}</div><div class=\"kv\"><b>geom_num_large_texture_regions</b>${r.geom_num_large_texture_regions??''}</div><div class=\"kv\"><b>geom_num_strong_boundaries</b>${r.geom_num_strong_boundaries??''}</div><div class=\"kv\"><b>geom_boundary_norm</b>${r.geom_boundary_norm??''}</div><div class=\"kv\"><b>stageB_clip_score</b>${r.stageB_clip_score??''}</div><div class=\"kv\"><b>stageD_score_0_100</b>${r.stageD_score_0_100??''}</div><div class=\"kv\"><b>stageA_n_masks</b>${r.stageA_n_masks??''}</div><div class=\"kv\"><b>stageA_largest_ratio</b>${r.stageA_largest_ratio??''}</div><div class=\"kv\"><b>stageA_small_frac</b>${r.stageA_small_frac??''}</div><div class=\"kv\"><b>stageA_pass</b>${r.stageA_pass}</div><div class=\"kv\"><b>stageB_pass</b>${r.stageB_pass}</div><div class=\"kv\"><b>stageD_decision</b>${r.stageD_decision??''}</div>`;el.detail.classList.add('open');document.getElementById('c').onclick=()=>el.detail.classList.remove('open');document.getElementById('p').onclick=()=>openDetail(Math.max(0,state.idx-1));document.getElementById('n').onclick=()=>openDetail(Math.min(state.filtered.length-1,state.idx+1));}
[el.dataset,el.status,el.sort,el.min,el.max,el.masks,el.obj,el.id].forEach(x=>x.addEventListener('input',filterRows));
el.reset.onclick=()=>{el.dataset.value='';el.status.value='';el.sort.value='final';el.min.value='0';el.max.value='100';el.masks.value='0';el.obj.value='1';el.id.value='';filterRows();};
document.addEventListener('keydown',e=>{if(!el.detail.classList.contains('open'))return;if(e.key==='Escape')el.detail.classList.remove('open');if(e.key==='ArrowLeft')openDetail(Math.max(0,state.idx-1));if(e.key==='ArrowRight')openDetail(Math.min(state.filtered.length-1,state.idx+1));});
initDatasets();
filterRows();
</script>
</body></html>"""
    (review_dir / "index.html").write_text(html, encoding="utf-8")
    return review_dir / "index.html"
