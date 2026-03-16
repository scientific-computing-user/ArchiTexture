#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
MANIFEST_PATH = ROOT / "table_manifest.json"
CI_EXPORT_PATH = SOURCE / "manuscript_exports" / "rwtd_confidence_summary.json"
DEFAULT_OUTPUT = ROOT / "generated" / "paper_tables.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt4(value: float | str) -> str:
    return f"{float(value):.4f}"


def fmt_pair(miou: float | str, ari: float | str) -> str:
    return f"{fmt4(miou)} / {fmt4(ari)}"


def fmt_pct1(value: float | str) -> str:
    return f"{100.0 * float(value):.1f}%"


def fmt_signed4(value: float | str) -> str:
    return f"{float(value):+.4f}"


def source_path(*parts: str) -> Path:
    return SOURCE.joinpath(*parts)


def manifest_tables() -> list[dict]:
    return load_json(MANIFEST_PATH)["tables"]


def expected_rows(table: dict) -> list[dict]:
    rows = table.get("rows")
    if rows is not None:
        return rows
    if table["build"] == "rwtd_confidence_export":
        return load_json(CI_EXPORT_PATH)["rows"]
    raise KeyError(f"Table {table['id']} has no expected rows")


def build_main_results(_: dict) -> list[dict]:
    rwtd_sam2 = load_json(source_path("main_results", "rwtd_sam2_original_official.json"))
    rwtd_texturesam = load_json(source_path("main_results", "rwtd_texturesam_common253_official.json"))
    rwtd_full = load_json(source_path("main_results", "rwtd_architexture_full256_official.json"))
    rwtd_common = load_json(source_path("main_results", "rwtd_architexture_common253_official.json"))
    stld_sam2 = load_json(source_path("main_results", "stld_sam2_original_summary.json"))
    stld_arch = load_json(source_path("main_results", "stld_architexture_summary.json"))
    stld_texturesam = load_json(source_path("main_results", "stld_texturesam_summary.json"))
    control_sam2 = load_json(source_path("main_results", "controlnet_sam2_original_summary.json"))
    control = load_json(source_path("main_results", "controlnet_partition_summary.json"))
    caid_sam2 = load_json(source_path("main_results", "caid_sam2_original_summary.json"))
    caid = load_json(source_path("main_results", "caid_partition_summary.json"))

    stld_sam2_row = stld_sam2["methods"]["SAM2_original"]
    stld_arch_row = stld_arch["methods"]["architexture"]
    stld_texturesam_row = stld_texturesam["methods"]["TextureSAM0p3"]
    control_sam2_row = control_sam2["methods"]["SAM2_original"]
    control_arch_row = control["methods"]["ArchiTexture_small_pre_ring_hi_0p3"]
    control_texturesam_row = control["methods"]["TextureSAM_0p3"]
    caid_sam2_row = caid_sam2["methods"]["SAM2_original"]
    caid_arch_row = caid["methods"]["ArchiTexture_small_pre_ring_hi_0p3"]
    caid_texturesam_row = caid["methods"]["TextureSAM_0p3"]

    return [
        {
            "group": "RWTD (natural route)",
            "Method": "SAM2.1-small original rerun",
            "Subset / coverage": f"full-256, {rwtd_sam2['noagg_official']['num_pred_image_ids']}/256",
            "mIoU": fmt4(rwtd_sam2["noagg_official"]["overall_average_iou"]),
            "ARI": fmt4(rwtd_sam2["noagg_official"]["overall_average_rand_index"]),
        },
        {
            "group": "RWTD (natural route)",
            "Method": "TextureSAM paper-reported RWTD result (eta <= 0.3)",
            "Subset / coverage": "paper-reported",
            "mIoU": "0.4700",
            "ARI": "0.6200",
        },
        {
            "group": "RWTD (natural route)",
            "Method": "ArchiTexture (ours)",
            "Subset / coverage": "full-256",
            "mIoU": fmt4(rwtd_full["noagg_official"]["overall_average_iou"]),
            "ARI": fmt4(rwtd_full["noagg_official"]["overall_average_rand_index"]),
        },
        {
            "group": "RWTD (natural route)",
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": "common-253",
            "mIoU": fmt4(rwtd_texturesam["noagg_official"]["overall_average_iou"]),
            "ARI": fmt4(rwtd_texturesam["noagg_official"]["overall_average_rand_index"]),
        },
        {
            "group": "RWTD (natural route)",
            "Method": "ArchiTexture (ours)",
            "Subset / coverage": "common-253",
            "mIoU": fmt4(rwtd_common["noagg_official"]["overall_average_iou"]),
            "ARI": fmt4(rwtd_common["noagg_official"]["overall_average_rand_index"]),
        },
        {
            "group": "STLD (controlled synthetic route)",
            "Method": "SAM2.1-small original rerun",
            "Subset / coverage": f"all-200, {stld_sam2_row['coverage']}/200",
            "mIoU": fmt4(stld_sam2_row["all"]["miou"]),
            "ARI": fmt4(stld_sam2_row["all"]["ari"]),
        },
        {
            "group": "STLD (controlled synthetic route)",
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": f"all-200, {stld_texturesam_row['coverage']}/200",
            "mIoU": fmt4(stld_texturesam_row["all"]["miou"]),
            "ARI": fmt4(stld_texturesam_row["all"]["ari"]),
        },
        {
            "group": "STLD (controlled synthetic route)",
            "Method": "ArchiTexture (ours)",
            "Subset / coverage": f"all-200, {stld_arch_row['all']['count']}/200",
            "mIoU": fmt4(stld_arch_row["all"]["miou"]),
            "ARI": fmt4(stld_arch_row["all"]["ari"]),
        },
        {
            "group": "STLD (controlled synthetic route)",
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": "common-182",
            "mIoU": fmt4(stld_texturesam_row["covered"]["miou"]),
            "ARI": fmt4(stld_texturesam_row["covered"]["ari"]),
        },
        {
            "group": "STLD (controlled synthetic route)",
            "Method": "ArchiTexture (ours)",
            "Subset / coverage": "common-182",
            "mIoU": fmt4(stld_arch_row["covered"]["miou"]),
            "ARI": fmt4(stld_arch_row["covered"]["ari"]),
        },
        {
            "group": "ControlNet-stitched PTD (generator-based bridge route)",
            "Method": "SAM2.1-small original rerun",
            "Subset / coverage": f"all-1742, {control_sam2_row['coverage']}/1742",
            "mIoU": fmt4(control_sam2_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(control_sam2_row["invariant"]["all"]["ari"]),
        },
        {
            "group": "ControlNet-stitched PTD (generator-based bridge route)",
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": f"all-1742, {control_texturesam_row['coverage']}/1742",
            "mIoU": fmt4(control_texturesam_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(control_texturesam_row["invariant"]["all"]["ari"]),
        },
        {
            "group": "ControlNet-stitched PTD (generator-based bridge route)",
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": f"all-1742, {control_arch_row['coverage']}/1742",
            "mIoU": fmt4(control_arch_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(control_arch_row["invariant"]["all"]["ari"]),
        },
        {
            "group": "CAID (domain-specific shoreline route)",
            "Method": "SAM2.1-small original rerun",
            "Subset / coverage": f"all-3104, {caid_sam2_row['coverage']}/3104",
            "mIoU": fmt4(caid_sam2_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(caid_sam2_row["invariant"]["all"]["ari"]),
        },
        {
            "group": "CAID (domain-specific shoreline route)",
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": f"all-3104, {caid_texturesam_row['coverage']}/3104",
            "mIoU": fmt4(caid_texturesam_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(caid_texturesam_row["invariant"]["all"]["ari"]),
        },
        {
            "group": "CAID (domain-specific shoreline route)",
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": f"all-3104, {caid_arch_row['coverage']}/3104",
            "mIoU": fmt4(caid_arch_row["invariant"]["all"]["miou"]),
            "ARI": fmt4(caid_arch_row["invariant"]["all"]["ari"]),
        },
    ]


def build_rwtd_main_baselines(_: dict) -> list[dict]:
    rows = {
        row["method"]: row
        for row in load_csv(source_path("rwtd_controls", "release_proposal_bank_baselines_summary.csv"))
    }
    full = load_json(source_path("main_results", "rwtd_architexture_full256_official.json"))
    order = [
        ("compatibility_only_core", "Core only"),
        ("repair_only_prob", "Repair only"),
        ("ranking_only_gain", "Rank only"),
        ("union_heuristic", "Union"),
        ("intersection_heuristic", "Intersection"),
        ("lightweight_reranker", "Lightweight reranker"),
    ]

    built = []
    for key, label in order:
        row = rows[key]
        built.append(
            {
                "Method": label,
                "mIoU": fmt4(row["full256_miou"]),
                "ARI": fmt4(row["full256_ari"]),
            }
        )
    built.append(
        {
            "Method": "ArchiTexture final",
            "mIoU": fmt4(full["noagg_official"]["overall_average_iou"]),
            "ARI": fmt4(full["noagg_official"]["overall_average_rand_index"]),
        }
    )
    return built


def build_proposal_source_swap(_: dict) -> list[dict]:
    rows = load_csv(source_path("rwtd_controls", "release_proposal_source_swap.csv"))
    label_map = {
        "released_dense_bank_official0p3": "Released dense bank",
        "sam21large_dense_bank_swap": "SAM2.1-large source swap",
    }
    return [
        {
            "Dense rescue proposal source (frozen)": label_map[row["proposal_source"]],
            "Full-256 mIoU / ARI": fmt_pair(row["full256_miou"], row["full256_ari"]),
            "Common-253 mIoU / ARI": fmt_pair(row["common253_miou"], row["common253_ari"]),
        }
        for row in rows
    ]


def build_controlnet_secondary(_: dict) -> list[dict]:
    summary = load_json(source_path("main_results", "controlnet_partition_summary.json"))
    texturesam = summary["methods"]["TextureSAM_0p3"]
    arch = summary["methods"]["ArchiTexture_small_pre_ring_hi_0p3"]
    return [
        {
            "Method": "TextureSAM 0.3 released checkpoint",
            "Subset / coverage": f"all-1742, {texturesam['coverage']}/1742",
            "Direct mIoU / ARI": fmt_pair(texturesam["direct"]["all"]["miou"], texturesam["direct"]["all"]["ari"]),
            "Invariant mIoU / ARI": fmt_pair(texturesam["invariant"]["all"]["miou"], texturesam["invariant"]["all"]["ari"]),
        },
        {
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": f"all-1742, {arch['coverage']}/1742",
            "Direct mIoU / ARI": fmt_pair(arch["direct"]["all"]["miou"], arch["direct"]["all"]["ari"]),
            "Invariant mIoU / ARI": fmt_pair(arch["invariant"]["all"]["miou"], arch["invariant"]["all"]["ari"]),
        },
        {
            "Method": "TextureSAM 0.3 released checkpoint",
            "Subset / coverage": "common-1739",
            "Direct mIoU / ARI": fmt_pair(texturesam["direct"]["covered"]["miou"], texturesam["direct"]["covered"]["ari"]),
            "Invariant mIoU / ARI": fmt_pair(texturesam["invariant"]["covered"]["miou"], texturesam["invariant"]["covered"]["ari"]),
        },
        {
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": "common-1739",
            "Direct mIoU / ARI": fmt_pair(arch["direct"]["covered"]["miou"], arch["direct"]["covered"]["ari"]),
            "Invariant mIoU / ARI": fmt_pair(arch["invariant"]["covered"]["miou"], arch["invariant"]["covered"]["ari"]),
        },
    ]


def build_caid_secondary(_: dict) -> list[dict]:
    summary = load_json(source_path("main_results", "caid_partition_summary.json"))
    texturesam = summary["methods"]["TextureSAM_0p3"]
    arch = summary["methods"]["ArchiTexture_small_pre_ring_hi_0p3"]
    return [
        {
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": f"all-3104, {texturesam['coverage']}/3104",
            "Invariant mIoU / ARI": fmt_pair(texturesam["invariant"]["all"]["miou"], texturesam["invariant"]["all"]["ari"]),
        },
        {
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": f"all-3104, {arch['coverage']}/3104",
            "Invariant mIoU / ARI": fmt_pair(arch["invariant"]["all"]["miou"], arch["invariant"]["all"]["ari"]),
        },
        {
            "Method": "TextureSAM public checkpoint rerun",
            "Subset / coverage": "common-3063",
            "Invariant mIoU / ARI": fmt_pair(texturesam["invariant"]["covered"]["miou"], texturesam["invariant"]["covered"]["ari"]),
        },
        {
            "Method": "ArchiTexture Stage-A (ours)",
            "Subset / coverage": "common-3063",
            "Invariant mIoU / ARI": fmt_pair(arch["invariant"]["covered"]["miou"], arch["invariant"]["covered"]["ari"]),
        },
    ]


def build_failure_audit(_: dict) -> list[dict]:
    audit = load_json(source_path("rwtd_audit", "audit_summary.json"))
    rescue = load_json(source_path("rwtd_audit", "rescue_reliability_summary.json"))
    wrong = load_json(source_path("rwtd_audit", "wrong_partition_summary.json"))
    per_image = load_csv(source_path("rwtd_audit", "release_per_image.csv"))
    topk_recoverable = sum(int(float(row["topk_oracle_recoverable"])) for row in per_image)

    return [
        {
            "Setting": "Baseline decision rule",
            "Resc.": str(int(rescue["rescue_activations"])),
            "Off-succ.": fmt_pct1(rescue["fraction_improve_official_pair"]),
            "Hurt-any": fmt_pct1(rescue["fraction_hurt_any_metric"]),
            "Wrong": str(int(wrong["wrong_partition_commitment_count"])),
            "Unsafe": str(int(audit["taxonomy_counts"]["unsafe_expansion"])),
            "Mean DeltaIoU": fmt_signed4(rescue["mean_iou_gain"]),
            "Top-k recov.": f"{topk_recoverable}/{audit['num_images']}",
        }
    ]


def build_rwtd_appendix_baselines(_: dict) -> list[dict]:
    rows = {
        row["method"]: row
        for row in load_csv(source_path("rwtd_controls", "release_proposal_bank_baselines_summary.csv"))
    }
    full = load_json(source_path("main_results", "rwtd_architexture_full256_official.json"))
    common = load_json(source_path("main_results", "rwtd_architexture_common253_official.json"))
    order = [
        ("compatibility_only_core", "Compatibility-only core", "conservative, consistency-preserving"),
        ("repair_only_prob", "Repair-only", "coverage-seeking, consistency drop"),
        ("ranking_only_gain", "Ranking-only", "severe partition errors"),
        ("union_heuristic", "Union heuristic", "over-merge collapse"),
        ("intersection_heuristic", "Intersection heuristic", "under-coverage collapse"),
        ("lightweight_reranker", "Lightweight reranker", "unstable tradeoff"),
    ]

    built = []
    for key, label, behavior in order:
        row = rows[key]
        built.append(
            {
                "Method": label,
                "Full-256 mIoU / ARI": fmt_pair(row["full256_miou"], row["full256_ari"]),
                "Common-253 mIoU / ARI": fmt_pair(row["common253_miou"], row["common253_ari"]),
                "Key behavior": behavior,
            }
        )
    built.append(
        {
            "Method": "ArchiTexture final",
            "Full-256 mIoU / ARI": fmt_pair(full["noagg_official"]["overall_average_iou"], full["noagg_official"]["overall_average_rand_index"]),
            "Common-253 mIoU / ARI": fmt_pair(common["noagg_official"]["overall_average_iou"], common["noagg_official"]["overall_average_rand_index"]),
            "Key behavior": "best consistency with competitive coverage",
        }
    )
    return built


def build_proposal_source_swap_appendix(_: dict) -> list[dict]:
    rows = load_csv(source_path("rwtd_controls", "release_proposal_source_swap.csv"))
    label_map = {
        "released_dense_bank_official0p3": "Released dense bank (official0.3 setup)",
        "sam21large_dense_bank_swap": "Source swap: SAM2.1-large bank",
    }
    notes_map = {
        "released_dense_bank_official0p3": "Main-paper release numbers",
        "sam21large_dense_bank_swap": "Same frozen consolidation weights; only dense rescue candidates change",
    }
    return [
        {
            "Dense rescue proposal source (frozen)": label_map[row["proposal_source"]],
            "Full-256 mIoU / ARI": fmt_pair(row["full256_miou"], row["full256_ari"]),
            "Common-253 mIoU / ARI": fmt_pair(row["common253_miou"], row["common253_ari"]),
            "Notes": notes_map[row["proposal_source"]],
        }
        for row in rows
    ]


def build_rwtd_confidence_export(_: dict) -> list[dict]:
    return load_json(CI_EXPORT_PATH)["rows"]


BUILDERS = {
    "manifest_rows": lambda table: table["rows"],
    "main_results": build_main_results,
    "rwtd_main_baselines": build_rwtd_main_baselines,
    "proposal_source_swap": build_proposal_source_swap,
    "controlnet_secondary": build_controlnet_secondary,
    "caid_secondary": build_caid_secondary,
    "failure_audit": build_failure_audit,
    "rwtd_appendix_baselines": build_rwtd_appendix_baselines,
    "proposal_source_swap_appendix": build_proposal_source_swap_appendix,
    "rwtd_confidence_export": build_rwtd_confidence_export,
}


def build_rows(table: dict) -> list[dict]:
    return BUILDERS[table["build"]](table)


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def render_markdown(tables_with_rows: list[tuple[dict, list[dict]]]) -> str:
    section_titles = {
        "main_paper": "Main Paper",
        "appendix": "Appendix",
    }
    lines = [
        "# ArchiTexture Paper Tables",
        "",
        "Generated from `paper_repro/table_manifest.json` and `paper_repro/source_data/`.",
        "",
        "Rebuild with `python paper_repro/scripts/build_paper_tables.py --check --write`.",
        "",
    ]
    current_section = None
    for table, rows in tables_with_rows:
        section = table["section"]
        if section != current_section:
            lines.extend([f"## {section_titles.get(section, section)}", ""])
            current_section = section

        lines.append(f"### `{table['id']}`")
        lines.append("")
        lines.append(table["title"])
        lines.append("")
        if table.get("notes"):
            lines.append(table["notes"])
            lines.append("")
        if table.get("public_sources"):
            lines.append("Public sources:")
            for ref in table["public_sources"]:
                lines.append(f"- `{ref}`")
            lines.append("")
        if table.get("workspace_sources"):
            lines.append("Original workspace sources:")
            for ref in table["workspace_sources"]:
                lines.append(f"- `{ref}`")
            lines.append("")

        headers = table["columns"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        current_group = None
        for row in rows:
            group = row.get("group")
            if group and group != current_group:
                lines.append("| " + " | ".join([f"**{escape_cell(group)}**"] + [""] * (len(headers) - 1)) + " |")
                current_group = group
            cells = [escape_cell(str(row.get(header, ""))) for header in headers]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def compare_rows(expected: list[dict], actual: list[dict]) -> bool:
    return expected == actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify the public ArchiTexture paper table bundle.")
    parser.add_argument("--check", action="store_true", help="Fail if rebuilt table rows drift from the committed manifest.")
    parser.add_argument("--write", action="store_true", help="Write the rendered markdown bundle.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown output path.")
    args = parser.parse_args()

    if not args.check and not args.write:
        args.check = True
        args.write = True

    manifest = manifest_tables()
    tables_with_rows: list[tuple[dict, list[dict]]] = []
    failures: list[str] = []

    for table in manifest:
        actual_rows = build_rows(table)
        expected = expected_rows(table)
        if args.check and not compare_rows(expected, actual_rows):
            failures.append(
                "\n".join(
                    [
                        f"[{table['id']}] row mismatch",
                        "Expected:",
                        json.dumps(expected, indent=2),
                        "Actual:",
                        json.dumps(actual_rows, indent=2),
                    ]
                )
            )
        tables_with_rows.append((table, actual_rows))

    if args.write:
        output_path = Path(args.output)
        output_path.write_text(render_markdown(tables_with_rows))

    if failures:
        sys.stderr.write("\n\n".join(failures) + "\n")
        return 1

    if args.check:
        print(f"Verified {len(tables_with_rows)} tables against the committed paper bundle.")
    if args.write:
        print(f"Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
