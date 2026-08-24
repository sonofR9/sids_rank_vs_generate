#!/usr/bin/env python3
"""Produce the revised scientific poster from the supplied PDF.

The source poster remains the base page.  Only the obsolete right-column plots
and the bottom "main result" band are removed from its content stream; the
overlay extends the existing experiment panel and adds the requested artwork.
"""

import csv
import json
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF = Path("/Users/konstantinryadinskiy/Downloads/poster_semantic_id_redesigned_updated.pdf")
OUTPUT_PDF = ROOT / "reports/poster/poster_semantic_id_updated.pdf"
RECALL_PNG = ROOT / "reports/figures/all_methods_sid_quality/recall100/recall100_vs_icr.png"
RANKING_PNG = ROOT / "reports/figures/all_methods_sid_quality/transact/transact_vs_intra_code_similarity.png"
RANKING_METADATA = ROOT / "reports/figures/ranking/analysis_metadata.json"
DCNV2_POINTS = ROOT / "reports/figures/all_methods_sid_quality/ranking_points.csv"
TRANSACT_POINTS = ROOT / "reports/figures/all_methods_sid_quality/transact_ranking_points.csv"
GENERATION_POINTS = ROOT / "reports/figures/all_methods_sid_quality/generation_points.csv"
SASREC_BASELINE = ROOT / "reports/experiments/sasrec/metrics/sasrec_yambda_50m_audio_uniform_seed42.json"

PAGE_WIDTH = 3370.394
PAGE_HEIGHT = 2383.937

# Coordinates match the existing right-hand panel in the supplied PDF.
PANEL_X = 2317.0
PANEL_WIDTH = 1017.394
PANEL_BOTTOM = 24.0
LOWER_PANEL_TOP = 1173.6
PLOT_X = 2346.0
PLOT_WIDTH = 959.3937
PLOT_HEIGHT = 423.7322
RECALL_Y = 739.5709
RANKING_Y = 100.0
TABLE_Y = 537.0
TABLE_HEIGHT = 166.0

PANEL_FILL = HexColor("#FFF5EA")
PANEL_BORDER = HexColor("#F4D5B3")
INK = HexColor("#102A4D")
ACCENT_RED = HexColor("#EF3340")
ACCENT_GREEN = HexColor("#2F7D4B")
CARD_BORDER = HexColor("#F3B2BA")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("PosterArial", "/System/Library/Fonts/Supplemental/Arial.ttf"))
    pdfmetrics.registerFont(TTFont("PosterArialBold", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))


def load_best_row(path: Path, metric: str) -> dict[str, str]:
    """Return the highest-metric SID result from an experiment point table."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get(metric)]
    if not rows:
        raise RuntimeError(f"No {metric} values found in {path}.")
    return max(rows, key=lambda row: float(row[metric]))


def sid_label(row: dict[str, str]) -> str:
    """Include the selected SID configuration without crowding the metric value."""
    method = row["method"]
    configuration = f"L={row['num_levels']}, C={row['codebook_size']}"
    return f"{method}, {configuration}" if method.endswith(")") else f"{method} ({configuration})"


def relative_change(best_metric: float, baseline_metric: float) -> float:
    return (best_metric - baseline_metric) / baseline_metric * 100


def table_rows() -> tuple[tuple[tuple[str, str], tuple[str, str], tuple[str, str], float], ...]:
    """Load all values used in the poster from their source experiment outputs."""
    with RANKING_METADATA.open(encoding="utf-8") as handle:
        ranking_metadata = json.load(handle)
    with SASREC_BASELINE.open(encoding="utf-8") as handle:
        sasrec_metrics = json.load(handle)

    dcnv2_baseline = float(ranking_metadata["ranking_baselines"]["dcnv2_pairwise_accuracy"])
    transact_baseline = float(ranking_metadata["ranking_baselines"]["transact_pairwise_accuracy"])
    generation_baseline = float(sasrec_metrics["metrics"]["recall"])

    dcnv2_best = load_best_row(DCNV2_POINTS, "dcnv2_pairwise_accuracy")
    transact_best = load_best_row(TRANSACT_POINTS, "transact_pairwise_accuracy")
    generation_best = load_best_row(GENERATION_POINTS, "recall_at_100")

    dcnv2_value = float(dcnv2_best["dcnv2_pairwise_accuracy"])
    transact_value = float(transact_best["transact_pairwise_accuracy"])
    generation_value = float(generation_best["recall_at_100"])

    return (
        (
            ("Ранжирование", "TransAct · Pairwise Accuracy"),
            ("TransAct item-only", f"{transact_baseline:.5f}"),
            (sid_label(transact_best), f"{transact_value:.5f}"),
            relative_change(transact_value, transact_baseline),
        ),
        (
            ("Ранжирование", "DCNv2 · Pairwise Accuracy"),
            ("DCNv2 (item-only)", f"{dcnv2_baseline:.5f}"),
            (sid_label(dcnv2_best), f"{dcnv2_value:.5f}"),
            relative_change(dcnv2_value, dcnv2_baseline),
        ),
        (
            ("Генерация", "Recall@100"),
            ("SASRec-Audio-Uniform", f"{generation_baseline:.5f}"),
            (sid_label(generation_best), f"{generation_value:.5f}"),
            relative_change(generation_value, generation_baseline),
        ),
    )


def rounded_rect_path(canvas: Canvas, x: float, y: float, width: float, height: float, radius: float):
    """Create a clip path matching ReportLab's rounded rectangle geometry."""
    kappa = 0.5522847498
    path = canvas.beginPath()
    path.moveTo(x + radius, y)
    path.lineTo(x + width - radius, y)
    path.curveTo(
        x + width - radius + kappa * radius,
        y,
        x + width,
        y + radius - kappa * radius,
        x + width,
        y + radius,
    )
    path.lineTo(x + width, y + height - radius)
    path.curveTo(
        x + width,
        y + height - radius + kappa * radius,
        x + width - radius + kappa * radius,
        y + height,
        x + width - radius,
        y + height,
    )
    path.lineTo(x + radius, y + height)
    path.curveTo(
        x + radius - kappa * radius,
        y + height,
        x,
        y + height - radius + kappa * radius,
        x,
        y + height - radius,
    )
    path.lineTo(x, y + radius)
    path.curveTo(x, y + radius - kappa * radius, x + radius - kappa * radius, y, x + radius, y)
    path.close()
    return path


def remove_obsolete_content(page) -> None:
    """Remove old graph image invocations and the entire bottom result band."""
    content = page.get_contents().get_data().decode("latin1")
    old_plots = (
        "q\n"
        "959.3937 0 0 423.7322 2346 739.5709 cm\n"
        "/FormXob.4247f64a1e1c8155a9ff1e2d497ffbf6 Do\n"
        "Q\n"
        "q\n"
        "959.3937 0 0 424.107 2346 304.3835 cm\n"
        "/FormXob.6c337b9bd7d62775e4088a7414c1bf41 Do\n"
        "Q\n"
    )
    if old_plots not in content:
        raise RuntimeError("Could not locate the two obsolete plot objects in the source poster.")
    content = content.replace(old_plots, "", 1)

    bottom_band = (
        "\n.043137 .192157 .352941 rg\n"
        ".043137 .192157 .352941 RG\n"
        "0 w\n"
        "n\n"
        "54 24 m\n"
    )
    band_offset = content.find(bottom_band)
    if band_offset == -1:
        raise RuntimeError("Could not locate the bottom main-result band in the source poster.")
    content = content[:band_offset] + "\n"

    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin1"))
    page[NameObject("/Contents")] = stream



def draw_lower_panel(canvas: Canvas) -> None:
    """Continue the original orange experiment panel to the poster bottom."""
    panel_top = LOWER_PANEL_TOP
    panel_right = PANEL_X + PANEL_WIDTH
    corner = 18.0

    canvas.setFillColor(PANEL_FILL)
    canvas.rect(PANEL_X, PANEL_BOTTOM, PANEL_WIDTH, panel_top - PANEL_BOTTOM, fill=1, stroke=0)

    # The source panel's side border already exists above panel_top.  Draw only
    # its continuation and the rounded lower edge, so the heatmap stays intact.
    path = canvas.beginPath()
    path.moveTo(PANEL_X, panel_top)
    path.lineTo(PANEL_X, PANEL_BOTTOM + corner)
    path.curveTo(
        PANEL_X,
        PANEL_BOTTOM + corner / 2,
        PANEL_X + corner / 2,
        PANEL_BOTTOM,
        PANEL_X + corner,
        PANEL_BOTTOM,
    )
    path.lineTo(panel_right - corner, PANEL_BOTTOM)
    path.curveTo(
        panel_right - corner / 2,
        PANEL_BOTTOM,
        panel_right,
        PANEL_BOTTOM + corner / 2,
        panel_right,
        PANEL_BOTTOM + corner,
    )
    path.lineTo(panel_right, panel_top)
    canvas.setStrokeColor(PANEL_BORDER)
    canvas.setLineWidth(1.3)
    canvas.drawPath(path, fill=0, stroke=1)


def draw_results_table(canvas: Canvas) -> None:
    """Draw the experiment summary between the two plot images."""
    table_x = PLOT_X
    table_width = PLOT_WIDTH
    header_height = 31.0
    corner_radius = 12.0
    column_widths = (190.0, 245.0, 350.0, 174.3937)
    headers = ("Задача", "Baseline", "Лучший SID", "Δ, %")
    rows = table_rows()
    row_height = (TABLE_HEIGHT - header_height) / len(rows)
    header_line = TABLE_Y + TABLE_HEIGHT - header_height

    # Every table layer is clipped to the same contour. This keeps the header
    # fill and grid inside the radiused card rather than merely drawing a
    # rounded border around a rectangular table.
    canvas.saveState()
    canvas.clipPath(
        rounded_rect_path(canvas, table_x, TABLE_Y, table_width, TABLE_HEIGHT, corner_radius),
        stroke=0,
        fill=0,
    )
    canvas.setFillColor(Color(1, 1, 1, alpha=0.96))
    canvas.rect(table_x, TABLE_Y, table_width, TABLE_HEIGHT, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#FDEEF1"))
    canvas.rect(table_x, header_line, table_width, header_height, fill=1, stroke=0)

    canvas.setStrokeColor(PANEL_BORDER)
    canvas.setLineWidth(0.7)
    canvas.line(table_x, header_line, table_x + table_width, header_line)
    for row_number in range(1, len(rows)):
        row_line = TABLE_Y + row_height * row_number
        canvas.line(table_x, row_line, table_x + table_width, row_line)

    x_positions = [table_x]
    for width in column_widths[:-1]:
        x_positions.append(x_positions[-1] + width)
    for x in x_positions[1:]:
        canvas.line(x, TABLE_Y, x, TABLE_Y + TABLE_HEIGHT)
    canvas.restoreState()

    # Stroke last so the continuous pink border stays visually clean above the
    # clipped grid and header fill.
    canvas.setStrokeColor(CARD_BORDER)
    canvas.setLineWidth(1.3)
    canvas.roundRect(table_x, TABLE_Y, table_width, TABLE_HEIGHT, corner_radius, fill=0, stroke=1)

    for x, width, heading in zip(x_positions, column_widths, headers):
        canvas.setFillColor(INK)
        canvas.setFont("PosterArialBold", 14)
        canvas.drawCentredString(x + width / 2, TABLE_Y + TABLE_HEIGHT - 20, heading)

    for row_index, row in enumerate(rows):
        row_bottom = TABLE_Y + row_height * (len(rows) - row_index - 1)
        text_y = row_bottom + 26
        small_y = row_bottom + 9

        label, baseline, best_sid, change = row
        canvas.setFillColor(INK)
        canvas.setFont("PosterArialBold", 14)
        canvas.drawString(x_positions[0] + 12, text_y, label[0])
        canvas.setFont("PosterArial", 11)
        canvas.drawString(x_positions[0] + 12, small_y, label[1])

        canvas.setFont("PosterArial", 12)
        canvas.drawString(x_positions[1] + 11, text_y, baseline[0])
        canvas.setFont("PosterArialBold", 16)
        canvas.drawString(x_positions[1] + 11, small_y, baseline[1])

        canvas.setFont("PosterArial", 12)
        canvas.drawString(x_positions[2] + 11, text_y, best_sid[0])
        canvas.setFont("PosterArialBold", 16)
        canvas.drawString(x_positions[2] + 11, small_y, best_sid[1])

        canvas.setFillColor(ACCENT_GREEN if change > 0 else ACCENT_RED)
        canvas.setFont("PosterArialBold", 18)
        canvas.drawCentredString(x_positions[3] + column_widths[3] / 2, row_bottom + 18, f"{change:+.1f}%")


def build_overlay() -> BytesIO:
    register_fonts()
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)

    draw_lower_panel(canvas)

    # Use the current, ready-made PNGs as supplied by the experiment reports.
    # Equal image ratio and placement dimensions preserve their aspect ratio.
    canvas.drawImage(
        ImageReader(str(RECALL_PNG)),
        PLOT_X,
        RECALL_Y,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )

    draw_results_table(canvas)

    canvas.drawImage(
        ImageReader(str(RANKING_PNG)),
        PLOT_X,
        RANKING_Y,
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )

    canvas.showPage()
    canvas.save()
    buffer.seek(0)
    return buffer


def main() -> None:
    for asset in (
        SOURCE_PDF,
        RECALL_PNG,
        RANKING_PNG,
        RANKING_METADATA,
        DCNV2_POINTS,
        TRANSACT_POINTS,
        GENERATION_POINTS,
        SASREC_BASELINE,
    ):
        if not asset.is_file():
            raise FileNotFoundError(asset)

    source = PdfReader(str(SOURCE_PDF))
    if len(source.pages) != 1:
        raise RuntimeError("Expected a single-page source poster.")
    page = source.pages[0]
    remove_obsolete_content(page)

    overlay = PdfReader(build_overlay())
    page.merge_page(overlay.pages[0], over=True)

    writer = PdfWriter()
    writer.add_page(page)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PDF.open("wb") as output:
        writer.write(output)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
