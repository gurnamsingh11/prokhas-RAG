# from docling.document_converter import DocumentConverter, ImageFormatOption
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.accelerator_options import AcceleratorOptions


# # Configure pipeline to run on CPU
# pipeline_options = PdfPipelineOptions(do_ocr=True)

# pipeline_options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=4)

# # Force OCR for images
# pipeline_options.ocr_options.force_full_page_ocr = True

# # Initialize converter
# converter = DocumentConverter(
#     allowed_formats=[InputFormat.IMAGE],
#     format_options={
#         InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)
#     },
# )


# def extract_text_from_image(image_path: str, converter=converter) -> str:
#     """
#     Extract text from an image using Docling OCR with CPU pipeline.

#     Parameters
#     ----------
#     image_path : str
#         Path to the image file.

#     Returns
#     -------
#     str
#         Extracted text from the image.
#     """
#     # Run conversion
#     result = converter.convert(image_path)

#     # Validate result
#     if result.document is None:
#         raise RuntimeError("Document conversion failed")

#     # Extract plain text
#     text = result.document.export_to_text()

#     return text.strip()


# ==============================================================

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── doctr singleton ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_doctr_model():
    """
    Load doctr OCR model once and cache it.
    Downloads ~130MB on first call, then reuses from cache.
    """
    from doctr.models import ocr_predictor

    logger.info("Loading doctr OCR model (first call downloads ~130MB)...")
    model = ocr_predictor(pretrained=True)
    logger.info("doctr model loaded.")
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────
# doctr works best on the original image (it does its own internal preprocessing)
# We only do minimal cleanup here


def _upscale_if_small(img: np.ndarray, min_height: int = 1000) -> np.ndarray:
    """Upscale if image is too small — doctr works better on larger images."""
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        logger.debug("Upscaled from %dx%d to %dx%d", w, h, img.shape[1], img.shape[0])
    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    """Fix tilted scans — even 2-3 degrees hurts OCR accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) > 0.3:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        logger.debug("Deskewed by %.2f degrees", angle)
    return img


def _preprocess(image_path: str) -> np.ndarray:
    """
    Minimal preprocessing for doctr.
    doctr handles contrast/noise internally — we only upscale and deskew.
    Heavy preprocessing (binarize, denoise) can actually hurt doctr's accuracy.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = _upscale_if_small(img)
    img = _deskew(img)
    return img


# ── Dynamic header cutoff ─────────────────────────────────────────────────────


def _detect_header_cutoff(img: np.ndarray):
    """
    Dynamically detect where the header ends and the table begins.

    Returns (cutoff_fraction, found) where:
      - cutoff_fraction : Y position as 0-1 fraction of image height
      - found           : True if a real line was detected, False if fallback

    This lets _assemble_text know whether to do zone splitting or not.
    If found=False → no table detected → don't split zones at all.
    """
    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_length = int(img_w * 0.30)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=20,
    )

    if lines is None:
        logger.debug("No lines found — no zone split")
        return 0.40, False

    y_fractions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 5:
            y_fractions.append(y1 / img_h)

    if not y_fractions:
        logger.debug("No horizontal lines — no zone split")
        return 0.40, False

    y_fractions.sort()
    for frac in y_fractions:
        if 0.10 < frac < 0.85:
            logger.debug("Header cutoff detected at %.1f%%", frac * 100)
            return frac, True

    logger.debug("No valid line — no zone split")
    return 0.40, False


# ── Column split detection ────────────────────────────────────────────────────


def _find_column_split(header_lines: List[Tuple], img: np.ndarray) -> float:
    """
    Find the X coordinate (0-1 fraction) that separates left/right columns.

    Only receives header lines (already filtered by _assemble_text).
    Looks for a horizontal gap in the middle of the page width.

    Returns split X fraction, or 0 if single-column layout.
    """
    if not header_lines:
        return 0

    x_centers = [x for (x, y, _) in header_lines]

    if len(x_centers) < 4:
        return 0

    # Bucket the X positions (20 buckets across page width)
    buckets = 20
    counts = [0] * buckets
    for x in x_centers:
        b = min(int(x * buckets), buckets - 1)
        counts[b] += 1

    # Find largest gap in middle 40% of page (buckets 4-16)
    max_gap = 0
    split_bucket = 0
    i = 4
    while i < 16:
        if counts[i] == 0:
            gap_len = 0
            j = i
            while j < 16 and counts[j] == 0:
                gap_len += 1
                j += 1
            if gap_len > max_gap:
                max_gap = gap_len
                split_bucket = i + gap_len // 2
            i = j
        else:
            i += 1

    if max_gap >= 1:
        split_x = split_bucket / buckets
        logger.debug("Column split at x=%.2f (%.0f%% of width)", split_x, split_x * 100)
        return split_x

    return 0


# ── doctr result parsing ──────────────────────────────────────────────────────


def _parse_doctr_result(result) -> List[Tuple[float, float, str]]:
    """
    Extract (x_center, y_center, text) from doctr result.

    doctr hierarchy: Page → Blocks → Lines → Words
    Coordinates are relative (0-1) so they work regardless of image size.

    We collect at LINE level (not word level) to preserve natural phrases
    like "CLEARANCE! Fast Dell Desktop" as one unit.
    """
    lines_with_pos = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                # Get line text by joining words
                text = " ".join(word.value for word in line.words).strip()
                if not text:
                    continue

                # Get line bounding box center (relative coordinates 0-1)
                geo = line.geometry  # ((x1,y1), (x2,y2))
                x_center = (geo[0][0] + geo[1][0]) / 2
                y_center = (geo[0][1] + geo[1][1]) / 2

                lines_with_pos.append((x_center, y_center, text))

    return lines_with_pos


# ── Text assembly ─────────────────────────────────────────────────────────────


def _assemble_text(lines_with_pos: List[Tuple], img: np.ndarray) -> str:
    """
    Assemble doctr lines into clean readable text.

    Three cases:
    1. No table detected (plain text image)
       → single column, sort top to bottom, done.

    2. Table detected, single column header
       → header top to bottom + table rows grouped left to right

    3. Table detected, two column header (Seller | Client)
       → header split into left/right + table rows grouped left to right
    """
    if not lines_with_pos:
        return ""

    # Check if image has a table border
    header_cutoff, table_found = _detect_header_cutoff(img)

    if not table_found:
        # ── Case 1: plain text — no zone split needed ─────────────────────────
        logger.debug("No table detected — single column assembly")
        lines_with_pos.sort(key=lambda l: l[1])
        return "\n".join(t for (_, _, t) in lines_with_pos)

    # ── Cases 2 & 3: table found — split into header and table zones ──────────
    header_lines = [(x, y, t) for (x, y, t) in lines_with_pos if y <= header_cutoff]
    table_lines = [(x, y, t) for (x, y, t) in lines_with_pos if y > header_cutoff]

    # ── Header zone: check for two columns ───────────────────────────────────
    header_text = ""
    if header_lines:
        split_x = _find_column_split(header_lines, img)
        if split_x > 0:
            left = sorted(
                [(x, y, t) for (x, y, t) in header_lines if x <= split_x],
                key=lambda l: l[1],
            )
            right = sorted(
                [(x, y, t) for (x, y, t) in header_lines if x > split_x],
                key=lambda l: l[1],
            )
            left_text = "\n".join(t for (_, _, t) in left)
            right_text = "\n".join(t for (_, _, t) in right)
            parts = []
            if left_text:
                parts.append(left_text)
            if right_text:
                parts.append(right_text)
            header_text = "\n\n---\n\n".join(parts)
        else:
            header_lines.sort(key=lambda l: l[1])
            header_text = "\n".join(t for (_, _, t) in header_lines)

    # ── Table zone: row-group then sort left-to-right within each row ─────────
    table_text = ""
    if table_lines:
        row_tolerance = 0.015
        table_lines.sort(key=lambda l: l[1])

        rows = []
        current_row = [table_lines[0]]
        current_y = table_lines[0][1]

        for line in table_lines[1:]:
            x, y, t = line
            if abs(y - current_y) <= row_tolerance:
                current_row.append(line)
            else:
                rows.append(current_row)
                current_row = [line]
                current_y = y
        rows.append(current_row)

        lines_out = []
        for row in rows:
            row.sort(key=lambda l: l[0])
            lines_out.append(" ".join(t for (_, _, t) in row))

        table_text = "\n".join(lines_out)

    # Join header and table
    sections = []
    if header_text:
        sections.append(header_text)
    if table_text:
        sections.append(table_text)
    return "\n\n".join(sections)


# ── Public function ───────────────────────────────────────────────────────────


def extract_text_from_image(image_path: str, image_name: str) -> Document:
    """
    Full OCR pipeline using doctr:
      preprocess → doctr OCR → parse lines → column-aware assembly → Document

    Returns one Document with do_not_split=True so SmartChunker keeps
    the whole image as a single chunk.
    """
    try:
        from doctr.io import DocumentFile

        # Step 1: load original image for line detection
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        # Step 2: preprocess for OCR (upscale + deskew only)
        img = _preprocess(image_path)

        # Step 2: save preprocessed image to temp file for doctr
        # doctr's DocumentFile.from_images accepts file paths
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, img)
        tmp.close()

        try:
            # Step 3: run doctr OCR
            model = _get_doctr_model()
            doc = DocumentFile.from_images(tmp.name)
            result = model(doc)
        finally:
            os.unlink(tmp.name)  # always clean up temp file

        # Step 4: parse doctr result into (x, y, text) tuples
        lines_with_pos = _parse_doctr_result(result)

        # Step 5: assemble with column awareness using original image for line detection
        text = _assemble_text(lines_with_pos, original)

        if not text.strip():
            logger.warning("doctr returned empty text for %s", image_name)
            text = "[No text extracted from image]"
        else:
            logger.info("doctr extracted %d chars from %s", len(text), image_name)

    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.error("OCR failed for %s: %s", image_name, exc)
        text = f"[OCR failed for {image_name}: {exc}]"

    return Document(
        page_content=text,
        metadata={
            "source": image_name,
            "page_label": "",
            "source_type": "image",
            "do_not_split": True,
        },
    )
