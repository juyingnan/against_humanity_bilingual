#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cah_bilingual.py

Utilities to:
1) Extract each black card's English text from a 3x3-per-page CAH-style PDF into CSV.
2) Render Chinese (and optional notes) back into the original PDF in a fixed textbox per card.

Key fix:
- Extraction uses word-level geometry (page.get_text("words")) instead of blocks, because some PDFs
  place multiple cards' text into a single "block" spanning multiple columns. Word geometry lets us
  correctly split text into the right card cell.

Dependencies:
  pip install pymupdf
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


GRID_ROWS = 3
GRID_COLS = 3

# These values match the provided cah-black.pdf (Letter: 612 x 792 points, 3x3 grid)
DEFAULT_LEFT_MARGIN = 36.0
DEFAULT_TOP_MARGIN = 24.0
DEFAULT_CARD_W = 180.0
DEFAULT_CARD_H = 248.4

MIDDOT_RE = re.compile(r"[·•・]")


@dataclass
class Card:
    card_id: str
    page: int  # 1-based
    row: int
    col: int
    en: str


def _replace_mid_dot(s: str) -> str:
    """Avoid CJK middle dot glyph issues by normalizing to ASCII dot."""
    return MIDDOT_RE.sub(".", s)


def _card_id(page_1based: int, row: int, col: int) -> str:
    return f"p{page_1based:02d}_r{row}_c{col}"


def _assign_cell(
    x_center: float,
    y_center: float,
    left_margin: float,
    top_margin: float,
    card_w: float,
    card_h: float,
) -> Optional[Tuple[int, int]]:
    """
    Given a point (x_center, y_center) in PyMuPDF page coordinates (origin top-left),
    return (row, col) for 3x3 grid, or None if outside the grid.
    """
    relx = x_center - left_margin
    rely = y_center - top_margin
    if relx < 0 or relx >= GRID_COLS * card_w:
        return None
    if rely < 0 or rely >= GRID_ROWS * card_h:
        return None
    col = int(relx // card_w)
    row = int(rely // card_h)
    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        return None
    return row, col


def _join_tokens(tokens: List[str]) -> str:
    """
    Join English tokens with spacing rules that avoid spaces before common punctuation.
    (PyMuPDF 'words' usually keeps punctuation attached, but this makes output more stable.)
    """
    if not tokens:
        return ""
    out: List[str] = [tokens[0]]
    no_space_before = {".", ",", "!", "?", ";", ":", ")", "]", "}"}
    no_space_after = {"(", "[", "{"}

    for tok in tokens[1:]:
        prev = out[-1]
        if tok in no_space_before:
            out[-1] = prev + tok
        elif tok.startswith("'") or tok.startswith("’"):
            out[-1] = prev + tok
        elif prev and prev[-1] in no_space_after:
            out[-1] = prev + tok
        else:
            out.append(tok)
    return " ".join(out)


def _words_to_lines(words: List[Tuple]) -> List[str]:
    """
    Convert a list of PyMuPDF 'words' into visual lines using y clustering.
    Each word tuple is (x0, y0, x1, y1, text, block_no, line_no, word_no).
    """
    if not words:
        return []

    # Sort top-to-bottom, left-to-right.
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))

    heights = [(w[3] - w[1]) for w in words_sorted]
    med_h = statistics.median(heights) if heights else 10.0
    # Tolerance: a fraction of font height, but not too small.
    y_tol = max(2.0, 0.25 * med_h)

    lines: List[List[Tuple]] = []
    cur: List[Tuple] = []
    cur_y: Optional[float] = None

    for w in words_sorted:
        y_center = (w[1] + w[3]) / 2.0
        if cur_y is None:
            cur = [w]
            cur_y = y_center
            continue
        if abs(y_center - cur_y) <= y_tol:
            cur.append(w)
            # Update running mean for stability.
            cur_y = (cur_y * (len(cur) - 1) + y_center) / len(cur)
        else:
            lines.append(cur)
            cur = [w]
            cur_y = y_center
    if cur:
        lines.append(cur)

    out_lines: List[str] = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w[0])
        toks = [w[4] for w in line_sorted]
        out_lines.append(_join_tokens(toks).strip())
    return [ln for ln in out_lines if ln]


def extract_cards(
    pdf_path: str,
    left_margin: float = DEFAULT_LEFT_MARGIN,
    top_margin: float = DEFAULT_TOP_MARGIN,
    card_w: float = DEFAULT_CARD_W,
    card_h: float = DEFAULT_CARD_H,
) -> List[Card]:
    """
    Extract cards using word-level geometry to avoid multi-card text blocks.
    """
    doc = fitz.open(pdf_path)
    cards: List[Card] = []

    for pno in range(len(doc)):
        page = doc[pno]
        words = page.get_text("words")  # per-word boxes

        # Group words into 3x3 cells.
        cell_words: Dict[Tuple[int, int], List[Tuple]] = {(r, c): [] for r in range(GRID_ROWS) for c in range(GRID_COLS)}
        for w in words:
            x0, y0, x1, y1, txt, *_ = w
            if not txt or not txt.strip():
                continue
            xc = (x0 + x1) / 2.0
            yc = (y0 + y1) / 2.0
            cell = _assign_cell(xc, yc, left_margin, top_margin, card_w, card_h)
            if cell is None:
                continue
            cell_words[cell].append(w)

        page_1based = pno + 1
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                lines = _words_to_lines(cell_words[(r, c)])
                en = "\n".join(lines).strip()
                cards.append(Card(card_id=_card_id(page_1based, r, c), page=page_1based, row=r, col=c, en=en))

    doc.close()
    return cards


def write_csv(cards: List[Card], out_csv: str) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["card_id", "page", "row", "col", "en", "zh", "note"])
        for c in cards:
            w.writerow([c.card_id, c.page, c.row, c.col, c.en, "", ""])


def _find_default_cjk_font() -> Optional[str]:
    """
    Best-effort font discovery for Windows/macOS/Linux. Users can (and should) pass --font explicitly.
    SimHei (黑体) is preferred for its bolder weight.
    """
    candidates = []
    if sys.platform.startswith("win"):
        win = os.environ.get("WINDIR", r"C:\Windows")
        candidates += [
            os.path.join(win, "Fonts", "simhei.ttf"),  # 黑体 (推荐，较粗)
            os.path.join(win, "Fonts", "NotoSansSC-VF.ttf"),  # Noto Sans SC
            os.path.join(win, "Fonts", "simfang.ttf"),  # 仿宋
        ]
    else:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def render_bilingual(
    pdf_in: str,
    csv_in: str,
    pdf_out: str,
    font_path: Optional[str] = None,
    left_margin: float = DEFAULT_LEFT_MARGIN,
    top_margin: float = DEFAULT_TOP_MARGIN,
    card_w: float = DEFAULT_CARD_W,
    card_h: float = DEFAULT_CARD_H,
    zh_fontsize: float = 8.0,
    note_fontsize: float = 7.0,
) -> None:
    """
    Insert zh/note into the original PDF. Uses fixed-width rectangles per card so text never
    spills into adjacent cards.
    """
    font_path = font_path or _find_default_cjk_font()
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError(
            "No CJK font found. Please pass --font with a valid .ttf/.ttc/.otf path "
            "(e.g., C:\\Windows\\Fonts\\simhei.ttf)."
        )

    # Read translations
    rows: Dict[str, Tuple[str, str]] = {}
    with open(csv_in, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = (r.get("card_id") or "").strip()
            if not cid:
                continue
            zh = (r.get("zh") or "").strip()
            note = (r.get("note") or "").strip()
            zh = _replace_mid_dot(zh)
            note = _replace_mid_dot(note)
            rows[cid] = (zh, note)

    doc = fitz.open(pdf_in)

    # Geometry for text areas inside each card (PyMuPDF coords: origin top-left)
    # Leave room at bottom for the CAH logo.
    inset_x = 12.0
    inset_top = 32.0
    bottom_logo_pad = 42.0  # reduced from 62 to push text lower

    zh_area_h = 36.0   # main Chinese translation
    zh_note_gap = 2.0   # gap between zh and note (reduced from implicit large gap)
    note_area_h = 26.0  # optional note area (increased for longer notes)
    line_height = 1.35  # line spacing multiplier
    
    # Maximum upward shift to avoid overflow (in points)
    max_upshift = 28.0
    upshift_step = 4.0
    
    def estimate_text_height(text: str, rect_width: float, fontsize: float) -> float:
        """Estimate text height based on character count and rect width."""
        if not text:
            return 0.0
        # Rough estimate: CJK chars ~= fontsize width, assume ~1.2x line height
        chars_per_line = max(1, int(rect_width / (fontsize * 0.9)))
        num_lines = (len(text) + chars_per_line - 1) // chars_per_line
        # Account for explicit newlines
        num_lines += text.count('\n')
        return num_lines * fontsize * 1.3

    for pno in range(len(doc)):
        page = doc[pno]
        page_1based = pno + 1

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cid = _card_id(page_1based, r, c)
                zh, note = rows.get(cid, ("", ""))
                if not zh and not note:
                    continue

                x0 = left_margin + c * card_w
                y0 = top_margin + r * card_h
                x1 = x0 + card_w
                y1 = y0 + card_h

                # Text box anchored above the logo area, near the lower part of the card.
                content_bottom = y1 - bottom_logo_pad
                rect_width = (x1 - inset_x) - (x0 + inset_x)
                
                # Calculate upshift needed based on text length estimation
                upshift = 0.0
                
                if note:
                    note_text = note if note.startswith("注：") or note.startswith("注:") else "注：" + note
                    est_note_h = estimate_text_height(note_text, rect_width, note_fontsize)
                    est_zh_h = estimate_text_height(zh, rect_width, zh_fontsize) if zh else 0
                    
                    # Check if estimated heights exceed allocated space
                    if est_note_h > note_area_h:
                        upshift = min(est_note_h - note_area_h + upshift_step, max_upshift)
                    if est_zh_h > zh_area_h:
                        upshift = min(max(upshift, est_zh_h - zh_area_h + upshift_step), max_upshift)
                else:
                    est_zh_h = estimate_text_height(zh, rect_width, zh_fontsize) if zh else 0
                    if est_zh_h > zh_area_h:
                        upshift = min(est_zh_h - zh_area_h + upshift_step, max_upshift)
                
                adjusted_bottom = content_bottom - upshift

                if note:
                    # Both zh and note: place note at bottom, zh directly above with small gap
                    note_rect = fitz.Rect(
                        x0 + inset_x,
                        adjusted_bottom - note_area_h,
                        x1 - inset_x,
                        adjusted_bottom,
                    )
                    zh_rect = fitz.Rect(
                        x0 + inset_x,
                        adjusted_bottom - note_area_h - zh_note_gap - zh_area_h,
                        x1 - inset_x,
                        adjusted_bottom - note_area_h - zh_note_gap,
                    )
                else:
                    # No note: place zh at the very bottom
                    zh_rect = fitz.Rect(
                        x0 + inset_x,
                        adjusted_bottom - zh_area_h,
                        x1 - inset_x,
                        adjusted_bottom,
                    )
                    note_rect = None

                # Insert main zh
                if zh:
                    rc = page.insert_textbox(
                        zh_rect,
                        zh,
                        fontname="cjk",
                        fontfile=font_path,
                        fontsize=zh_fontsize,
                        align=0,  # left
                        lineheight=line_height,
                    )
                    if rc < 0:
                        if upshift > 0:
                            print(f"[WARN] zh overflow: {cid} (even with {upshift:.1f}pt upshift)", file=sys.stderr)
                        else:
                            print(f"[WARN] zh overflow: {cid}", file=sys.stderr)
                    elif upshift > 0:
                        print(f"[INFO] {cid}: shifted up by {upshift:.1f}pt", file=sys.stderr)

                # Insert note
                if note and note_rect:
                    rc = page.insert_textbox(
                        note_rect,
                        note_text,
                        fontname="cjk",
                        fontfile=font_path,
                        fontsize=note_fontsize,
                        align=0,
                        lineheight=line_height,
                    )
                    if rc < 0:
                        print(f"[WARN] note overflow: {cid}", file=sys.stderr)

    # Save with compression and garbage collection to reduce file size
    doc.save(
        pdf_out,
        garbage=4,  # Maximum garbage collection
        deflate=True,  # Compress streams
        clean=True,  # Clean content streams
    )
    doc.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ex = sub.add_parser("extract", help="Extract English card texts into a CSV.")
    ap_ex.add_argument("--pdf", required=True)
    ap_ex.add_argument("--out", required=True)
    ap_ex.add_argument("--left", type=float, default=DEFAULT_LEFT_MARGIN)
    ap_ex.add_argument("--top", type=float, default=DEFAULT_TOP_MARGIN)
    ap_ex.add_argument("--card_w", type=float, default=DEFAULT_CARD_W)
    ap_ex.add_argument("--card_h", type=float, default=DEFAULT_CARD_H)

    ap_re = sub.add_parser("render", help="Insert zh/note into the original PDF.")
    ap_re.add_argument("--pdf", required=True)
    ap_re.add_argument("--csv", required=True)
    ap_re.add_argument("--out", required=True)
    ap_re.add_argument("--font", default=None, help="Path to a CJK font file (recommended).")
    ap_re.add_argument("--left", type=float, default=DEFAULT_LEFT_MARGIN)
    ap_re.add_argument("--top", type=float, default=DEFAULT_TOP_MARGIN)
    ap_re.add_argument("--card_w", type=float, default=DEFAULT_CARD_W)
    ap_re.add_argument("--card_h", type=float, default=DEFAULT_CARD_H)
    ap_re.add_argument("--zh_fontsize", type=float, default=8.0)
    ap_re.add_argument("--note_fontsize", type=float, default=7.0)

    args = ap.parse_args()

    if args.cmd == "extract":
        cards = extract_cards(
            args.pdf,
            left_margin=args.left,
            top_margin=args.top,
            card_w=args.card_w,
            card_h=args.card_h,
        )
        write_csv(cards, args.out)
        print(f"Wrote {len(cards)} cards to {args.out}")

    elif args.cmd == "render":
        render_bilingual(
            args.pdf,
            args.csv,
            args.out,
            font_path=args.font,
            left_margin=args.left,
            top_margin=args.top,
            card_w=args.card_w,
            card_h=args.card_h,
            zh_fontsize=args.zh_fontsize,
            note_fontsize=args.note_fontsize,
        )
        print(f"Wrote bilingual PDF to {args.out}")

    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
