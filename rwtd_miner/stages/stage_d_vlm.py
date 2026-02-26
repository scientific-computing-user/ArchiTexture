from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from rwtd_miner.utils.logging import get_logger


@dataclass
class VLMScore:
    score_0_100: int | None
    decision: str | None
    main_reason: str | None
    flags: list[str]


class BaseVLMScorer:
    def score_image_vlm(self, image_path: Path) -> VLMScore:
        raise NotImplementedError


class StubVLMScorer(BaseVLMScorer):
    def score_image_vlm(self, image_path: Path) -> VLMScore:
        _ = image_path
        return VLMScore(score_0_100=None, decision=None, main_reason=None, flags=[])


class ExternalCommandVLMScorer(BaseVLMScorer):
    def __init__(self, command: str, prompt: str):
        self.command = command
        self.prompt = prompt

    def score_image_vlm(self, image_path: Path) -> VLMScore:
        payload = {"image_path": str(image_path), "prompt": self.prompt}
        proc = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"External VLM command failed: {proc.stderr.strip()}")

        raw = proc.stdout.strip()
        if not raw:
            raise RuntimeError("External VLM command returned empty output")
        parsed = json.loads(raw)

        return VLMScore(
            score_0_100=int(parsed.get("score_0_100")) if parsed.get("score_0_100") is not None else None,
            decision=str(parsed.get("decision")) if parsed.get("decision") is not None else None,
            main_reason=str(parsed.get("main_reason")) if parsed.get("main_reason") is not None else None,
            flags=[str(x) for x in parsed.get("flags", [])] if isinstance(parsed.get("flags", []), list) else [],
        )


def _resolve_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def _parse_yes_no(text: str) -> bool | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    yes_terms = ("yes", "y", "true", "mostly yes", "definitely yes")
    no_terms = ("no", "n", "false", "mostly no", "definitely no")
    if any(t == x or t.startswith(f"{x} ") for x in yes_terms):
        return True
    if any(t == x or t.startswith(f"{x} ") for x in no_terms):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def _parse_region_count(text: str) -> int | None:
    t = (text or "").strip().lower()
    if not t:
        return None
    m = re.search(r"\d+", t)
    if m:
        return int(m.group(0))
    word_map = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    for w, v in word_map.items():
        if w in t:
            return v
    return None


class HfBlipVqaRubricScorer(BaseVLMScorer):
    def __init__(self, cfg: dict[str, Any]):
        self.log = get_logger("stage_d_blip_vqa")
        self.cfg = cfg
        self.device = _resolve_device(str(cfg.get("device_preference", "auto")))
        self.model_name = str(cfg.get("hf_blip_vqa_model_name", "Salesforce/blip-vqa-base"))
        self.max_new_tokens = int(cfg.get("hf_blip_max_new_tokens", 12))

        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.log.info("Stage D backend: hf_blip_vqa (%s) on %s", self.model_name, self.device)

    @torch.no_grad()
    def _ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return str(text).strip().lower()

    def score_image_vlm(self, image_path: Path) -> VLMScore:
        with Image.open(image_path) as im:
            image = im.convert("RGB")

        q_real = "Is this a real natural photograph (not collage, not graphic design)? Answer yes or no."
        q_texture = "Are most pixels textured surfaces or materials rather than distinct objects? Answer yes or no."
        q_boundary = "Is there a clear boundary or transition between two or more texture/material regions? Answer yes or no."
        q_object = "Is the image dominated by a single salient object such as person, car, animal, or product? Answer yes or no."
        q_regions = "How many large texture/material regions are visible? answer a number from 1 to 5."

        real_ans = self._ask(image, q_real)
        texture_ans = self._ask(image, q_texture)
        boundary_ans = self._ask(image, q_boundary)
        object_ans = self._ask(image, q_object)
        regions_ans = self._ask(image, q_regions)

        is_real = _parse_yes_no(real_ans)
        is_texture_dom = _parse_yes_no(texture_ans)
        has_boundary = _parse_yes_no(boundary_ans)
        object_dominant = _parse_yes_no(object_ans)
        n_regions = _parse_region_count(regions_ans)

        flags: list[str] = []
        score = 50.0
        reasons: list[str] = []

        if is_real is True:
            score += 10.0
            reasons.append("real photo")
        elif is_real is False:
            score -= 35.0
            flags.extend(["mosaic_or_collage", "synthetic_or_graphic"])
            reasons.append("not real photo")

        if is_texture_dom is True:
            score += 22.0
            reasons.append("texture-dominant")
        elif is_texture_dom is False:
            score -= 24.0
            flags.append("low_texture_content")
            reasons.append("low texture dominance")

        if has_boundary is True:
            score += 20.0
            reasons.append("clear texture boundary")
        elif has_boundary is False:
            score -= 20.0
            flags.append("no_clear_texture_boundary")
            reasons.append("weak boundary")

        if object_dominant is True:
            score -= 30.0
            flags.append("object_centric")
            reasons.append("object-centric")
        elif object_dominant is False:
            score += 10.0
            reasons.append("not object-centric")

        if n_regions is not None:
            if 2 <= n_regions <= 4:
                score += 12.0
                reasons.append(f"{n_regions} large regions")
            elif n_regions <= 1:
                score -= 10.0
            else:
                score -= 8.0
                flags.append("too_many_objects")

        score_i = int(max(0, min(100, round(score))))
        if score_i >= 80:
            decision = "match"
        elif score_i >= 60:
            decision = "borderline"
        else:
            decision = "not_match"

        # Keep flags stable and unique.
        dedup_flags: list[str] = []
        for f in flags:
            if f not in dedup_flags:
                dedup_flags.append(f)

        reason = ", ".join(reasons[:3]) if reasons else "insufficient VLM signal"
        return VLMScore(score_0_100=score_i, decision=decision, main_reason=reason, flags=dedup_flags)


def build_vlm_scorer(cfg: dict[str, Any]) -> BaseVLMScorer:
    backend = str(cfg.get("backend", "stub"))
    if backend == "stub":
        return StubVLMScorer()
    if backend == "external_command":
        command = str(cfg.get("external_command", "")).strip()
        if not command:
            raise ValueError("stage_d.external_command is required for external_command backend")
        prompt = str(cfg.get("prompt", ""))
        return ExternalCommandVLMScorer(command=command, prompt=prompt)
    if backend == "hf_blip_vqa":
        return HfBlipVqaRubricScorer(cfg=cfg)
    raise ValueError(f"Unsupported stage_d backend: {backend}")


def run_stage_d(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    log = get_logger("stage_d")
    out = df.copy()
    out["stageD_score_0_100"] = None
    out["stageD_decision"] = None
    out["stageD_flags"] = None
    out["stageD_reason"] = None
    out["stageD_error"] = None

    scorer = build_vlm_scorer(cfg)

    top_n = int(cfg.get("score_top_n_from_stage_b", 2000))
    shortlist = out[out["stageB_pass"].fillna(False)].copy()
    shortlist = shortlist.sort_values("stageB_clip_score", ascending=False).head(top_n)

    if shortlist.empty:
        return out

    for idx, row in tqdm(shortlist.iterrows(), total=len(shortlist), desc="stageD_vlm", unit="img"):
        image_path = Path(str(row["image_path"]))
        try:
            result = scorer.score_image_vlm(image_path=image_path)
            out.at[idx, "stageD_score_0_100"] = result.score_0_100
            out.at[idx, "stageD_decision"] = result.decision
            out.at[idx, "stageD_flags"] = "|".join(result.flags)
            out.at[idx, "stageD_reason"] = result.main_reason
        except Exception as exc:
            out.at[idx, "stageD_error"] = str(exc)
            log.warning("VLM scoring failed for %s: %s", image_path, exc)

    return out
