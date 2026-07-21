"""Build a labelled contact sheet from rendered manuscript pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=300)
    args = parser.parse_args()

    pages = sorted(args.input.glob("page-*.png"))
    if not pages:
        raise SystemExit("No rendered pages found")
    images = []
    for path in pages:
        source = Image.open(path).convert("RGB")
        height = round(source.height * args.thumb_width / source.width)
        images.append(source.resize((args.thumb_width, height), Image.Resampling.LANCZOS))
    label_height = 24
    gap = 14
    rows = (len(images) + args.columns - 1) // args.columns
    cell_height = max(image.height for image in images) + label_height
    sheet = Image.new(
        "RGB",
        (
            args.columns * args.thumb_width + (args.columns + 1) * gap,
            rows * cell_height + (rows + 1) * gap,
        ),
        "#D9D9D9",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, image in enumerate(images):
        row, column = divmod(index, args.columns)
        x = gap + column * (args.thumb_width + gap)
        y = gap + row * (cell_height + gap)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 2), f"Page {index + 1}", fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=94)


if __name__ == "__main__":
    main()

