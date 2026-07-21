#!/usr/bin/env python3
"""Extract DKAN.pdf into a traceable draft reader bundle.

The extractor deliberately keeps source text and page anchors separate from any
later human/LLM translation.  It is deterministic so that block IDs remain
stable across follow-up research passes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


HEADING_RE = re.compile(
    r"^(?:Abstract|References|Acknowledg(?:e)?ments?|Appendix(?:\s+[A-Z])?|"
    r"\d+(?:\.\d+)*\.?\s+[A-Z].{0,110})$"
)
CAPTION_RE = re.compile(r"^(?:Fig\.|Figure\s|Table\s|Algorithm\s)", re.I)
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")


def clean_lines(text: str) -> list[str]:
    """Normalize extraction artifacts while preserving mathematical content."""
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        if line.startswith("arXiv:"):
            continue
        cleaned.append(line)
    return cleaned


def join_wrapped(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and line and line[0].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return re.sub(r"\s+", " ", out).strip()


def split_blocks(text: str) -> list[dict[str, str]]:
    lines = clean_lines(text)
    blocks: list[dict[str, str]] = []
    current: list[str] = []

    def flush(kind: str = "paragraph") -> None:
        nonlocal current
        value = join_wrapped(current)
        if value:
            blocks.append({"type": kind, "text": value})
        current = []

    for line in lines:
        if not line:
            flush()
            continue
        if CAPTION_RE.match(line):
            flush()
            blocks.append({"type": "caption", "text": line})
            continue
        if (HEADING_RE.match(line) or (len(line) <= 52 and line in {"Abstract", "References"})):
            flush()
            blocks.append({"type": "heading", "text": line})
            continue
        current.append(line)
        if len(join_wrapped(current)) >= 1400 and re.search(r"[.!?;:]$", line):
            flush()
    flush()
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(args.input_pdf))
    metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items()}

    block_no = 0
    caption_no = 0
    order = 0
    blocks: list[dict] = []
    pages: list[dict] = []
    raw_pages: list[dict] = []

    for page_index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        raw_pages.append({"page": page_index, "text": raw})
        page_ids: list[str] = []
        for item in split_blocks(raw):
            order += 1
            if item["type"] == "caption":
                caption_no += 1
                block_id = f"C{caption_no:03d}"
            else:
                block_no += 1
                block_id = f"S{block_no:03d}"
            page_ids.append(block_id)
            blocks.append(
                {
                    "id": block_id,
                    "page": page_index,
                    "type": item["type"],
                    "order": order,
                    "original_text": item["text"],
                    "translation": "",
                    "bbox": [],
                    "confidence": "medium",
                    "refs": [],
                    "insert_after": "",
                }
            )
        pages.append({"page": page_index, "block_ids": page_ids})

    title = metadata.get("/Title", "Discontinuity-aware KAN-based physics-informed neural networks")
    source_map = {
        "paper": {
            "title": title,
            "authors": metadata.get("/Author", ""),
            "venue": "arXiv",
            "identifier": metadata.get("/arXivID", metadata.get("/DOI", "")),
            "source_type": "pdf",
            "language": "en",
            "source_path": str(args.input_pdf.resolve()),
            "page_count": len(reader.pages),
        },
        "blocks": blocks,
        "pages": pages,
        "figures": [],
        "glossary": [],
    }

    (args.output_dir / "source_map.json").write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "raw_pages.json").write_text(
        json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "raw_text.txt").write_text(
        "\n\n".join(f"===== PAGE {p['page']} =====\n{p['text']}" for p in raw_pages),
        encoding="utf-8",
    )

    md: list[str] = [
        f"# {title}",
        "",
        "> Draft source-grounded reader. Original blocks are complete; Chinese translation and figure cards are pending review.",
        "",
        f"- Authors: {metadata.get('/Author', '')}",
        f"- Source: {metadata.get('/arXivID', metadata.get('/DOI', ''))}",
        f"- Pages: {len(reader.pages)}",
        "",
        "## Page index",
        "",
        " | ".join(f"[p.{p['page']}](#page-{p['page']})" for p in pages),
        "",
        "## Terminology ledger (draft)",
        "",
        "| Canonical term | Chinese | Decision |",
        "|---|---|---|",
        "| DPINN | 不连续性感知物理信息神经网络 | Use for the complete method |",
        "| DKAN | 不连续性感知 Kolmogorov-Arnold 网络 | Use for the core network architecture |",
        "| learnable local artificial viscosity | 可学习局部人工粘性 | Keep distinct from fixed/global AV |",
        "| adaptive Fourier-feature embedding | 自适应 Fourier 特征嵌入 | Preserve Fourier capitalization |",
        "",
    ]
    for page in pages:
        md.extend([f'<a id="page-{page["page"]}"></a>', f"## Page {page['page']}", ""])
        for block_id in page["block_ids"]:
            block = next(b for b in blocks if b["id"] == block_id)
            if block["type"] == "heading":
                md.extend([f'<a id="{block_id}"></a>', f"### {block['original_text']}", ""])
                continue
            md.extend(
                [
                    f'<a id="{block_id}"></a>',
                    f"**Source:** p.{block['page']} {block_id}",
                    "",
                    f"**Original:** {block['original_text']}",
                    "",
                    "**中文:** 【待译；当前为全文锚点草稿，禁止据此视为已完成翻译。】",
                    "",
                ]
            )
    (args.output_dir / "paper.md").write_text("\n".join(md), encoding="utf-8")

    notes = """# Translation and extraction notes

- Status: draft mode.
- The selectable text layer was extracted for all pages with deterministic page/block IDs.
- Chinese translation is intentionally left pending rather than silently summarized.
- Bounding boxes and figure/table cards are pending visual crop verification.
- Multi-column reading order has medium confidence and must be checked against rendered pages.
"""
    (args.output_dir / "translation_notes.md").write_text(notes, encoding="utf-8")

    print(
        json.dumps(
            {
                "pages": len(reader.pages),
                "blocks": len(blocks),
                "captions": caption_no,
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
