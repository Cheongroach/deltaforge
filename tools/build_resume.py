# SPDX-License-Identifier: Apache-2.0
"""Render RESUME.md to PDF without rewriting the authored Markdown."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def normalize(text):
    for char in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad":
        text = text.replace(char, "-")
    return text


def rich(text):
    # Project-relative Markdown links become clickable GitHub references in PDF.
    pieces = []
    cursor = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        pieces.append(escape(normalize(text[cursor:match.start()])))
        label, target = match.groups()
        if not target.startswith(("https://", "http://")):
            target = "https://github.com/CheongRoach/deltaforge/blob/main/" + target
        pieces.append(f'<link href="{escape(target, {chr(34): "&quot;"})}" '
                      f'color="#2b6cb0">{escape(normalize(label))}</link>')
        cursor = match.end()
    pieces.append(escape(normalize(text[cursor:])))
    return "".join(pieces)


def build(source, output, font):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    PAGE_SIZE = A4
    PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE
    LEFT_MARGIN = RIGHT_MARGIN = 40
    TOP_MARGIN = BOTTOM_MARGIN = 35
    CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
    CONTENT_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    pdfmetrics.registerFont(TTFont("CJKFont", str(font), subfontIndex=0))
    pdfmetrics.registerFontFamily("CJKFont", normal="CJKFont", bold="CJKFont",
                                 italic="CJKFont", boldItalic="CJKFont")
    # Compact resume adaptations of the professional PDF palette and styles.
    common = dict(fontName="CJKFont", wordWrap="CJK", textColor=HexColor("#2d3748"))
    styles = {
        "title": ParagraphStyle("Title", fontName="CJKFont", wordWrap="CJK",
                               fontSize=24, leading=29, textColor=HexColor("#1a365d"),
                               spaceAfter=10, keepWithNext=True),
        "body": ParagraphStyle("Body", **common, fontSize=10, leading=15, spaceAfter=7),
        "h1": ParagraphStyle("H1", fontName="CJKFont", wordWrap="CJK", fontSize=13,
                            leading=19, textColor=HexColor("#1a365d"), spaceBefore=11,
                            spaceAfter=7, keepWithNext=True),
        "cell": ParagraphStyle("Cell", **common, fontSize=9.5, leading=14),
        "th": ParagraphStyle("TH", fontName="CJKFont", wordWrap="CJK", fontSize=9.5,
                            leading=14, textColor=HexColor("#ffffff")),
    }
    def p(text, kind="body"):
        return Paragraph(rich(text), styles[kind])

    lines = source.read_text().splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table = Table([[p(c, "th" if n == 0 else "cell") for c in row]
                           for n, row in enumerate(rows)],
                          colWidths=[CONTENT_WIDTH * .39, CONTENT_WIDTH * .61],
                          repeatRows=1)
            table.setStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2b6cb0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [HexColor("#f7fafc"), HexColor("#ffffff")]),
                ("LINEBELOW", (0, 0), (-1, -1), .5, HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
            story.extend([table, Spacer(1, 7)])
            continue
        if line.startswith("# "):
            story.append(p(line[2:], "title"))
        elif line.startswith("## "):
            story.append(p(line[3:], "h1"))
        elif line:
            story.append(p("• " + line[2:] if line.startswith("- ") else line))
        i += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=PAGE_SIZE,
                            leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
                            topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
                            title="AI Infra 项目简历 | DeltaForge", author="")

    def footer(canvas, document):
        canvas.setStrokeColor(HexColor("#e2e8f0"))
        canvas.line(LEFT_MARGIN, 25, PAGE_WIDTH - RIGHT_MARGIN, 25)
        canvas.setFont("CJKFont", 8)
        canvas.setFillColor(HexColor("#718096"))
        canvas.drawString(LEFT_MARGIN, 14, normalize("DeltaForge · 2026.10 · 项目简历"))
        canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 14, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {output}; content area {CONTENT_WIDTH:.1f}×{CONTENT_HEIGHT:.1f} pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "RESUME.md")
    parser.add_argument("--output", type=Path, default=ROOT / "RESUME.pdf")
    parser.add_argument("--font", type=Path,
                        default=ROOT / "profiling-lab/_shared/fonts/wqy-microhei.ttc")
    args = parser.parse_args()
    if not args.source.is_file() or not args.font.is_file():
        parser.error("Source Markdown and CJK font must exist")
    build(args.source, args.output, args.font)
