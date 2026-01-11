#!/usr/bin/env python3
"""
cah_bilingual.py
Make bilingual Cards Against Humanity-style PDFs by adding Chinese text onto the existing card PDF.

Workflow:
  1) Extract card texts (English) to CSV:
       python cah_bilingual.py extract --pdf cah-black.pdf --out cards.csv
  2) Fill in 'zh' and optionally 'note' columns in the CSV.
  3) Render a new bilingual PDF (non-destructive: writes a new file):
       python cah_bilingual.py render --pdf cah-black.pdf --csv cards.csv --out cah-black-bilingual.pdf

Notes:
  - This script assumes the PDF is a 3x3 grid per page (like your cah-black.pdf example).
  - It preserves the original layout and only adds text.
  - Chinese text is inserted using PyMuPDF (fitz) with an embedded font file (recommended).
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF


# --- Layout constants for cah-black.pdf (Letter, 3x3 grid) ---
CARD_W = 180.0
CARD_H = 248.4
LEFT_MARGIN = 36.0
TOP_MARGIN = 24.0  # from top edge downward


@dataclass
class Card:
    card_id: str
    page: int
    row: int
    col: int
    en: str


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _normalize_cn(s: str) -> str:
    """
    CN normalization:
      - remove optional leading "中文："/"中文:"
      - replace middle dots with '.' to avoid missing-glyph issues
    """
    s = (s or "").strip()
    s = re.sub(r"^\s*中文[:：]\s*", "", s)
    # Normalize various middle dot / bullet characters.
    s = s.replace("·", ".").replace("•", ".").replace("・", ".")
    return s.strip()


def _card_rect_from_grid_fitz(row: int, col: int) -> fitz.Rect:
    """
    Return card rectangle in PyMuPDF coordinates (origin at top-left).
    """
    x0 = LEFT_MARGIN + col * CARD_W
    y0 = TOP_MARGIN + row * CARD_H
    return fitz.Rect(x0, y0, x0 + CARD_W, y0 + CARD_H)


def _auto_find_cn_font() -> Optional[str]:
    """
    Try to find a usable CJK sans-serif font file on the current machine.
    Prefer TTF/OTF. TTC may work with PyMuPDF, but availability varies by OS.
    """
    candidates = []

    # Linux (container): Noto Sans CJK is commonly installed as TTC
    candidates += [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf",
    ]

    # Windows
    candidates += [
        r"C:\Windows\Fonts\msyh.ttc",     # Microsoft YaHei (TTC)
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",   # SimHei (TTF)
        r"C:\Windows\Fonts\msyhl.ttc",
    ]

    # macOS
    candidates += [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def extract_cards_to_csv(pdf_path: str, out_csv: str) -> None:
    """
    Extract English prompt text per card to CSV.
    Uses text blocks and clusters them into a 3x3 grid.
    """
    doc = fitz.open(pdf_path)
    cards: List[Card] = []

    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("blocks")

        filtered = []
        for (x0, y0, x1, y1, text, bno, btype) in blocks:
            t = _normalize_text(text)
            if not t:
                continue
            if "Cards Against Humanity is a trademark" in t:
                continue
            filtered.append((x0, y0, x1, y1, t))

        # Sort blocks into reading order
        filtered.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

        # If the PDF matches expected grid, we can just read each card's main textbox.
        # However, to stay close to the original demo, we map blocks to a 3x3 grid
        # using x0/y0 buckets.
        x0s = [round(b[0], 1) for b in filtered]
        y0s = [round(b[1], 1) for b in filtered]

        def _top3_buckets(vals: List[float]) -> List[float]:
            freq: Dict[float, int] = {}
            for v in vals:
                # bucket in 5-pt bins
                k = round(v / 5) * 5
                freq[k] = freq.get(k, 0) + 1
            tops = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            return sorted([k for k, _ in tops])

        x_buckets = _top3_buckets(x0s)
        y_buckets = _top3_buckets(y0s)

        def _nearest_bucket(v: float, buckets: List[float]) -> int:
            return min(range(len(buckets)), key=lambda i: abs(v - buckets[i]))

        grid_text = {(r, c): [] for r in range(3) for c in range(3)}
        for (x0, y0, x1, y1, t) in filtered:
            r = _nearest_bucket(round(y0 / 5) * 5, y_buckets)
            c = _nearest_bucket(round(x0 / 5) * 5, x_buckets)
            grid_text[(r, c)].append(t)

        for r in range(3):
            for c in range(3):
                en = "\n".join(grid_text[(r, c)]).strip()
                if not en:
                    continue
                card_id = f"p{pno+1:02d}_r{r}_c{c}"
                cards.append(Card(card_id=card_id, page=pno + 1, row=r, col=c, en=en))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "page", "row", "col", "en", "zh", "note"])
        for c in cards:
            w.writerow([c.card_id, c.page, c.row, c.col, c.en, "", ""])


def render_bilingual_pdf(pdf_path: str, cards_csv: str, out_pdf: str, fontfile: Optional[str]) -> None:
    """
    Add Chinese text onto the existing PDF and save as a new PDF.
    """
    # Load translations
    rows: Dict[str, Dict[str, str]] = {}
    with open(cards_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows[row["card_id"]] = row

    # Pick font
    if fontfile:
        cn_fontfile = fontfile
    else:
        cn_fontfile = _auto_find_cn_font()

    if not cn_fontfile:
        raise RuntimeError(
            "No CJK font file found. Provide one with --font, e.g. "
            "Windows: C:\\Windows\\Fonts\\simhei.ttf or msyh.ttc; "
            "macOS: /System/Library/Fonts/PingFang.ttc; "
            "Linux: install Noto Sans CJK."
        )

    doc = fitz.open(pdf_path)

    for pno in range(len(doc)):
        page = doc[pno]
        for r in range(3):
            for c in range(3):
                card_id = f"p{pno+1:02d}_r{r}_c{c}"
                if card_id not in rows:
                    continue

                zh = _normalize_cn(rows[card_id].get("zh") or "")
                note = _normalize_cn(rows[card_id].get("note") or "")
                if not zh and not note:
                    continue

                card = _card_rect_from_grid_fitz(r, c)

                # In-card text boxes (keep clear of the logo area at the bottom)
                left = card.x0 + 12
                right = card.x1 - 12
                bottom_pad = 36.0  # leave space for logo
                note_h = 22.0
                zh_h = 44.0

                note_rect = fitz.Rect(left, card.y1 - bottom_pad - note_h, right, card.y1 - bottom_pad)
                zh_rect = fitz.Rect(left, note_rect.y0 - zh_h, right, note_rect.y0)

                # Chinese main translation (max space = zh_rect; auto-wrap inside box)
                if zh:
                    ret = page.insert_textbox(
                        zh_rect,
                        zh,
                        fontsize=8,
                        fontname="CNFont",
                        fontfile=cn_fontfile,
                        align=0,
                    )
                    if ret < 0:
                        print(f"[WARN] zh text overflow: {card_id}", file=os.sys.stderr)

                # Optional note (smaller)
                if note:
                    ret2 = page.insert_textbox(
                        note_rect,
                        "注：" + note,
                        fontsize=7,
                        fontname="CNFont",
                        fontfile=cn_fontfile,
                        align=0,
                    )
                    if ret2 < 0:
                        print(f"[WARN] note text overflow: {card_id}", file=os.sys.stderr)

    doc.save(out_pdf, garbage=4, deflate=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ex = sub.add_parser("extract", help="Extract English card texts to CSV")
    ap_ex.add_argument("--pdf", required=True)
    ap_ex.add_argument("--out", required=True)

    ap_re = sub.add_parser("render", help="Render bilingual PDF from CSV")
    ap_re.add_argument("--pdf", required=True)
    ap_re.add_argument("--csv", required=True)
    ap_re.add_argument("--out", required=True)
    ap_re.add_argument(
        "--font",
        default=None,
        help="Path to a CJK font file (TTF/OTF recommended; TTC often works). "
             "If omitted, the script tries to auto-detect a font.",
    )

    args = ap.parse_args()

    if args.cmd == "extract":
        extract_cards_to_csv(args.pdf, args.out)
    elif args.cmd == "render":
        render_bilingual_pdf(args.pdf, args.csv, args.out, args.font)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
