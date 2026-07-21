"""Audit manuscript figure exports and build a Python-only contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw


EXPECTED = [
    "fig1_architecture",
    "fig2_dkan_mechanism",
    "fig3_constraint_logic",
    "fig12_mechanism_suite_overview",
    "fig13_mechanism_sod",
    "fig14_mechanism_lax",
    "fig15_mechanism_moving_contact",
    "fig16_mechanism_double_rarefaction",
    "fig17_mechanism_colliding_streams",
    "fig18_mechanism_shu_osher",
    "fig19_mechanism_woodward_colella",
    "fig20_mechanism_smooth_entropy",
    "fig21_sensor_ablation",
    "fig22_reference_convergence",
    "fig11_multiseed_ablation",
    "fig10_engineering_benchmark",
    "fig12_tv_trust_diagnostic",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    figures = Path(args.figures)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    records = []
    previews = []
    for name in EXPECTED:
        paths = {extension: figures / f"{name}.{extension}" for extension in ("svg", "pdf", "png", "tiff")}
        missing = [extension for extension, path in paths.items() if not path.exists()]
        png = Image.open(paths["png"])
        tiff = Image.open(paths["tiff"])
        svg_root = ET.parse(paths["svg"]).getroot()
        text_nodes = [node for node in svg_root.iter() if node.tag.endswith("text")]
        record = {
            "name": name,
            "missing": missing,
            "png_pixels": list(png.size),
            "tiff_pixels": list(tiff.size),
            "tiff_dpi": [float(value) for value in tiff.info.get("dpi", (0, 0))],
            "svg_text_nodes": len(text_nodes),
            "pdf_bytes": paths["pdf"].stat().st_size,
            "passed": not missing and png.width >= 1800 and tiff.width >= 3600 and len(text_nodes) > 0,
        }
        records.append(record)
        preview = png.convert("RGB")
        preview.thumbnail((720, 640), Image.Resampling.LANCZOS)
        previews.append((name, preview.copy()))

    cell_width, cell_height = 760, 700
    rows = (len(previews) + 1) // 2
    sheet = Image.new("RGB", (2 * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, preview) in enumerate(previews):
        column, row = index % 2, index // 2
        x = column * cell_width + (cell_width - preview.width) // 2
        y = row * cell_height + 38 + (cell_height - 58 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((column * cell_width + 18, row * cell_height + 12), name, fill="black")
    sheet.save(output / "figure_contact_sheet.png", dpi=(150, 150))

    report = {
        "expected_count": len(EXPECTED),
        "passed_count": sum(record["passed"] for record in records),
        "all_passed": all(record["passed"] for record in records),
        "records": records,
    }
    (output / "figure_qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
