"""Build a PDF contact sheet and report page-level rendering diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader


def ink_fraction(image: Image.Image) -> float:
    grey = image.convert("L")
    mask = grey.point(lambda value: 255 if value < 245 else 0)
    bbox = ImageChops.difference(mask, Image.new("L", mask.size, 0)).getbbox()
    if bbox is None:
        return 0.0
    histogram = mask.histogram()
    return histogram[255] / (image.width * image.height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pdf = Path(args.pdf)
    render_dir = Path(args.render_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pages = sorted(render_dir.glob("page-*.png"))
    reader = PdfReader(pdf)
    if len(pages) != len(reader.pages):
        raise RuntimeError(f"rendered {len(pages)} pages for a {len(reader.pages)}-page PDF")

    records = []
    previews = []
    for index, path in enumerate(pages, start=1):
        image = Image.open(path).convert("RGB")
        page = reader.pages[index - 1]
        text = page.extract_text() or ""
        records.append(
            {
                "page": index,
                "pixels": list(image.size),
                "ink_fraction": ink_fraction(image),
                "extracted_characters": len(text),
                "media_box_points": [float(page.mediabox.width), float(page.mediabox.height)],
            }
        )
        preview = image.copy()
        preview.thumbnail((390, 520), Image.Resampling.LANCZOS)
        previews.append((index, preview))

    columns = 4
    rows = (len(previews) + columns - 1) // columns
    cell_width, cell_height = 420, 570
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for offset, (page_number, preview) in enumerate(previews):
        column, row = offset % columns, offset // columns
        x = column * cell_width + (cell_width - preview.width) // 2
        y = row * cell_height + 30 + (cell_height - 45 - preview.height) // 2
        sheet.paste(preview, (x, y))
        draw.text((column * cell_width + 12, row * cell_height + 9), f"page {page_number}", fill="black")
    sheet.save(output / "manuscript_contact_sheet.png", dpi=(150, 150))

    report = {
        "pdf": str(pdf),
        "page_count": len(reader.pages),
        "minimum_ink_fraction": min(item["ink_fraction"] for item in records),
        "maximum_ink_fraction": max(item["ink_fraction"] for item in records),
        "blank_pages": [item["page"] for item in records if item["ink_fraction"] < 0.005],
        "records": records,
    }
    (output / "pdf_qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
