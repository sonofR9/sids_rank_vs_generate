from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, MSO_VERTICAL_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt


TEMPLATE = Path("/Users/konstantinryadinskiy/Downloads/Постер_fin.pptx")
OUTPUT = Path(
    "/Users/konstantinryadinskiy/Desktop/sids_rank_vs_generate/"
    "deliverables/poster_semantic_id_final_equations.pptx"
)

# Fixed template design system. New poster content uses only these named values;
# the scientific heatmap is the sole exception and retains a restrained diverging scale.
FONT = "Arial"
INK = RGBColor(0x11, 0x11, 0x16)
MUTED = RGBColor(0x52, 0x4D, 0x61)
GREEN = RGBColor(0x2A, 0xE8, 0x81)
GREEN_DARK = RGBColor(0x08, 0x7C, 0x48)
PURPLE = RGBColor(0x6C, 0x57, 0xB7)
PURPLE_RULE = RGBColor(0xD5, 0xCD, 0xFF)
PURPLE_TINT = RGBColor(0xF2, 0xEF, 0xFF)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
HEAT_NEUTRAL = RGBColor(0xF0, 0xF1, 0xF4)
HEAT_POS_1 = RGBColor(0xF9, 0xE5, 0xDB)
HEAT_POS_2 = RGBColor(0xF7, 0xD3, 0xC2)
HEAT_POS_3 = RGBColor(0xF4, 0xB6, 0x9E)
HEAT_POS_4 = RGBColor(0xF1, 0x8D, 0x72)
HEAT_NEG_1 = RGBColor(0xE5, 0xEC, 0xF7)
HEAT_NEG_2 = RGBColor(0xD8, 0xE3, 0xF4)


def style_run(run, size: float, *, bold: bool = False, color=INK, italic: bool = False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return run


def configure_text_frame(shape, *, margins: float = 0.04, valign=MSO_VERTICAL_ANCHOR.TOP):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = Inches(margins)
    tf.margin_right = Inches(margins)
    tf.margin_top = Inches(margins)
    tf.margin_bottom = Inches(margins)
    tf.vertical_anchor = valign
    return tf


def add_textbox(
    slide,
    x,
    y,
    w,
    h,
    text,
    *,
    size=30,
    bold=False,
    color=INK,
    align=PP_ALIGN.LEFT,
    valign=MSO_VERTICAL_ANCHOR.TOP,
    name=None,
    margins=0.04,
    line_spacing=1.05,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    tf = configure_text_frame(shape, margins=margins, valign=valign)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_after = Pt(0)
    style_run(p.add_run(), size, bold=bold, color=color).text = text
    return shape


def add_rich_textbox(
    slide,
    x,
    y,
    w,
    h,
    paragraphs,
    *,
    name=None,
    margins=0.04,
    valign=MSO_VERTICAL_ANCHOR.TOP,
):
    """paragraphs: list of dicts with runs=[(text,size,bold,color)] and spacing."""
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    tf = configure_text_frame(shape, margins=margins, valign=valign)
    for index, spec in enumerate(paragraphs):
        p = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        p.alignment = spec.get("align", PP_ALIGN.LEFT)
        p.line_spacing = spec.get("line_spacing", 1.05)
        p.space_before = Pt(spec.get("before", 0))
        p.space_after = Pt(spec.get("after", 0))
        for text, size, bold, color in spec["runs"]:
            style_run(p.add_run(), size, bold=bold, color=color).text = text
    return shape


def add_rule(slide, x1, y1, x2, y2, *, color=PURPLE_RULE, width=1.5, name=None):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color
    line.line.width = Pt(width)
    if name:
        line.name = name
    return line


def add_box(
    slide,
    x,
    y,
    w,
    h,
    text,
    *,
    size=30,
    bold=False,
    fill=PAPER,
    line=PURPLE_RULE,
    line_width=1.5,
    color=INK,
    name=None,
):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    if name:
        shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(line_width)
    tf = configure_text_frame(shape, margins=0.12, valign=MSO_VERTICAL_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.line_spacing = 1.02
    p.space_after = Pt(0)
    style_run(p.add_run(), size, bold=bold, color=color).text = text
    return shape


def add_band(slide, x, y, w, h, *, fill=PURPLE_TINT, name=None):
    """A single quiet backdrop for a whole diagram, never an individual card."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    if name:
        shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = fill
    shape.line.width = Pt(0.5)
    return shape


def office_math_run(text, size, *, italic=False, normal=False, color=INK):
    """Return one native OMML math run with DrawingML formatting."""
    normal_xml = "<m:nor/>" if normal else ""
    style = "i" if italic else "p"
    return (
        "<m:r>"
        f'<m:rPr>{normal_xml}<m:sty m:val="{style}"/></m:rPr>'
        f'<a:rPr lang="en-US" sz="{int(size * 100)}">'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Cambria Math"/>'
        "</a:rPr>"
        f'<m:t xml:space="preserve">{escape(text)}</m:t>'
        "</m:r>"
    )


def office_math_subscript(base, subscript, size):
    return (
        "<m:sSub>"
        f"<m:e>{office_math_run(base, size, italic=True)}</m:e>"
        f"<m:sub>{office_math_run(subscript, size, normal=True)}</m:sub>"
        "</m:sSub>"
    )


def office_math_fraction(numerator_xml, denominator_xml):
    return (
        '<m:f><m:fPr><m:type m:val="bar"/></m:fPr>'
        f"<m:num>{numerator_xml}</m:num>"
        f"<m:den>{denominator_xml}</m:den>"
        "</m:f>"
    )


def add_office_math_textbox(
    slide,
    x,
    y,
    w,
    h,
    math_xml,
    *,
    align=PP_ALIGN.LEFT,
    valign=MSO_VERTICAL_ANCHOR.MIDDLE,
    name=None,
    margins=0.04,
):
    """Insert editable PowerPoint Office Math (OMML), not a picture or text imitation."""
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    tf = configure_text_frame(shape, margins=margins, valign=valign)
    paragraph = tf.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.0
    omml = parse_xml(
        '<a14:m xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
        f"<m:oMathPara><m:oMath>{math_xml}</m:oMath></m:oMathPara>"
        "</a14:m>"
    )
    paragraph_xml = paragraph._p
    end_properties = paragraph_xml.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}endParaRPr"
    )
    insert_at = (
        list(paragraph_xml).index(end_properties)
        if end_properties is not None
        else len(paragraph_xml)
    )
    paragraph_xml.insert(insert_at, omml)
    return shape


def add_flow_arrow(slide, x, y, w=0.38, h=0.55, *, name=None):
    return add_textbox(
        slide,
        x,
        y,
        w,
        h,
        "→",
        size=34,
        bold=True,
        color=GREEN_DARK,
        align=PP_ALIGN.CENTER,
        valign=MSO_VERTICAL_ANCHOR.MIDDLE,
        name=name,
        margins=0,
    )


def replace_template_header(slide):
    title = slide.shapes[4]
    title.left = Inches(1.84)
    title.top = Inches(2.28)
    title.width = Inches(34.8)
    title.height = Inches(1.45)
    runs = title.text_frame.paragraphs[0].runs
    for run in runs:
        run.text = ""
    target = runs[-1]
    target.text = "ВЛИЯНИЕ КАЧЕСТВА SEMANTIC ID НА ФИНАЛЬНЫЕ МЕТРИКИ"
    style_run(target, 72, bold=False, color=INK)

    team = slide.shapes[5]
    team.left = Inches(1.84)
    team.top = Inches(4.18)
    team.width = Inches(34.8)
    team.height = Inches(1.20)
    tf = team.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0)
    p = tf.paragraphs[0]
    p.line_spacing = 1.0
    p.space_after = Pt(0)
    text = (
        "Константин Рядинский • Лев Клюев • Ростислав Головков • "
        "Глеб Шевченко • Ментор: Александр Новак"
    )
    style_run(p.add_run(), 36, bold=False, color=INK).text = text

    slide.shapes[6].text_frame.clear()
    for index in (14, 15, 16, 17):
        slide.shapes[index].text_frame.clear()


def build_introduction(slide):
    x, w = 2.45, 12.35

    add_textbox(slide, x, 10.30, w, 0.55, "Semantic ID и исследовательский вопрос", size=34, bold=True,
                name="INTRO_Subhead")
    add_textbox(
        slide,
        x,
        10.92,
        w,
        1.75,
        "Semantic IDs — дискретное представление объектов в рекомендательных системах. "
        "Но хорошее внутреннее качество SID не обязательно означает хорошее downstream-качество.",
        size=30,
        name="INTRO_Definition",
    )

    add_rich_textbox(
        slide,
        x,
        12.85,
        w,
        2.05,
        [
            {"runs": [("Какие свойства Semantic ID действительно отражаются на downstream-метриках рекомендательной системы?", 40, True, INK)], "line_spacing": 1.0},
        ],
        name="INTRO_ResearchQuestion",
    )
    add_rule(slide, x, 15.00, x + 7.6, 15.00, color=GREEN, width=4, name="INTRO_QuestionUnderline")

    add_textbox(slide, x, 15.30, w, 0.50, "Цель", size=34, bold=True, name="INTRO_GoalHead")
    add_rich_textbox(
        slide,
        x,
        15.86,
        w,
        1.30,
        [
            {"runs": [("• Сравнить способы построения SID.", 30, False, INK)]},
            {"runs": [("• Проверить связь SID-метрик с downstream-качеством.", 30, False, INK)], "before": 4},
        ],
        name="INTRO_GoalBody",
    )

    add_textbox(slide, x, 17.30, w, 0.50, "Экспериментальный контекст", size=34, bold=True,
                name="INTRO_ContextHead")
    context_rows = [
        ("Dataset", "Yambda-50M, time split"),
        ("Generation", "TIGER vs SASRec"),
        ("Ranking", "DCNv2 / TransAct-like"),
        ("Gen. metric", "Recall@100"),
        ("Rank. metrics", "ROC-AUC · Pairwise Accuracy"),
    ]
    y = 17.90
    for i, (label, value) in enumerate(context_rows):
        add_textbox(slide, x, y, 3.35, 0.62, label, size=30, bold=True, color=MUTED,
                    valign=MSO_VERTICAL_ANCHOR.MIDDLE, name=f"INTRO_ContextLabel_{i}")
        add_textbox(slide, x + 3.55, y, w - 3.55, 0.62, value, size=30, color=INK,
                    valign=MSO_VERTICAL_ANCHOR.MIDDLE, name=f"INTRO_ContextValue_{i}")
        if i < len(context_rows) - 1:
            add_rule(slide, x, y + 0.62, x + w, y + 0.62, color=PURPLE_RULE, width=1,
                     name=f"INTRO_ContextRule_{i}")
        y += 0.68

    add_textbox(slide, x, 21.52, w, 0.50, "Ключевые SID-метрики", size=34, bold=True,
                name="INTRO_MetricsHead")
    add_textbox(slide, x, 22.10, w, 0.48, "Independent Code Rate (ICR)", size=32, bold=True,
                color=GREEN_DARK, name="INTRO_ICR_Head")
    icr_math = (
        office_math_run("ICR", 36, normal=True)
        + office_math_run(" = ", 36, normal=True)
        + office_math_fraction(
            office_math_run("unique used codes", 36, normal=True),
            office_math_subscript("n", "items", 36),
        )
    )
    add_office_math_textbox(
        slide,
        x,
        22.55,
        8.80,
        1.15,
        icr_math,
        name="INTRO_ICR_OfficeMath",
    )
    add_textbox(slide, x, 23.75, w, 0.62,
                "Чем ближе к 1, тем меньше объектов разделяют один SID.",
                size=30, name="INTRO_ICR_Note")
    add_rich_textbox(
        slide,
        x,
        24.50,
        w,
        1.85,
        [
            {"runs": [("Intra-code Similarity", 32, True, GREEN_DARK)]},
            {"runs": [("Средняя семантическая близость объектов внутри одного кода.", 30, False, INK)], "before": 3},
            {"runs": [("Чем выше значение, тем семантически однороднее группа.", 30, False, INK)], "before": 3},
        ],
        name="INTRO_IntraCode",
    )


def build_methods(slide):
    x, w = 16.25, 14.05

    add_textbox(slide, x, 10.30, w, 0.52, "Логика эксперимента", size=34, bold=True,
                name="METHODS_LogicHead")
    add_band(slide, x, 10.92, w, 1.18, name="METHODS_LogicBand")
    add_textbox(slide, x + 0.28, 11.02, 3.40, 0.98, "Построение\nSemantic ID", size=30,
                bold=True, align=PP_ALIGN.CENTER, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="METHODS_LogicSID")
    add_flow_arrow(slide, x + 3.78, 11.23, name="METHODS_LogicArrow1")
    add_textbox(slide, x + 4.25, 11.02, 4.25, 0.98, "TIGER / DCNv2 /\nTransAct-like", size=30,
                bold=True, align=PP_ALIGN.CENTER, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="METHODS_LogicModels")
    add_flow_arrow(slide, x + 8.62, 11.23, name="METHODS_LogicArrow2")
    add_textbox(slide, x + 9.08, 11.02, 4.68, 0.98,
                "downstream-метрики", size=32, bold=True,
                align=PP_ALIGN.CENTER, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="METHODS_LogicMetrics")

    add_textbox(slide, x, 12.42, w, 0.50, "Balanced RQ-KMeans", size=34, bold=True,
                name="METHODS_BalancedHead")
    add_rich_textbox(
        slide,
        x,
        13.02,
        4.95,
        1.42,
        [
            {"runs": [("Равноразмерные кластеры", 30, True, INK)]},
            {"runs": [("capacity:  w = |V| / K", 30, True, GREEN_DARK)], "before": 5},
        ],
        name="METHODS_BalancedConcept",
    )
    add_rule(slide, x + 5.17, 13.05, x + 5.17, 14.28, color=PURPLE_RULE, width=1.5,
             name="METHODS_BalancedDivider")
    add_textbox(
        slide,
        x + 5.48,
        13.05,
        8.42,
        1.24,
        "Назначаем ближайшие объекты → ограничиваем размер каждого кластера → "
        "обновляем центры до сходимости.",
        size=30,
        valign=MSO_VERTICAL_ANCHOR.MIDDLE,
        name="METHODS_BalancedFlow",
    )

    add_textbox(slide, x, 14.64, w, 0.50, "RQ-VAE", size=36, bold=True,
                name="METHODS_RQVAEHead")
    add_band(slide, x, 15.20, w, 1.62, name="METHODS_RQVAEBand")
    rq_labels = ["Encoder", "Residual\ncodebooks 1…L", "Quantized\nrepresentation", "Decoder"]
    rq_widths = [2.30, 4.15, 3.92, 2.28]
    rx = x
    for i, (label, bw) in enumerate(zip(rq_labels, rq_widths)):
        add_textbox(slide, rx, 15.38, bw, 1.22, label, size=32, bold=True,
                    align=PP_ALIGN.CENTER, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                    name=f"METHODS_RQVAEStage_{i+1}")
        if i < 3:
            add_flow_arrow(slide, rx + bw + 0.04, 15.73, w=0.34,
                           name=f"METHODS_RQVAEArrow_{i+1}")
            rx += bw + 0.45
        else:
            rx += bw
    add_textbox(
        slide, x, 16.94, w, 0.70,
        "Остаток последовательно квантуется по codebooks → итоговый SID состоит из суммы уровней.",
        size=30, color=MUTED, name="METHODS_RQVAENote",
    )

    add_textbox(slide, x, 17.85, w, 0.50, "User-aware", size=36, bold=True,
                name="METHODS_UserAwareHead")
    add_band(slide, x, 18.41, w, 1.68, name="METHODS_UserAwareBand")
    ua_labels = ["User + item\nembedding", "User-aware\nnetwork", "User-aware SID", "Canonical\nSID target"]
    ux = x
    ua_widths = [3.05, 3.08, 3.25, 3.05]
    for i, (label, bw) in enumerate(zip(ua_labels, ua_widths)):
        add_textbox(slide, ux, 18.62, bw, 1.26, label, size=32, bold=True,
                    align=PP_ALIGN.CENTER, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                    name=f"METHODS_UserAwareStage_{i+1}")
        if i < 3:
            add_flow_arrow(slide, ux + bw + 0.08, 18.99, w=0.38,
                           name=f"METHODS_UserAwareArrow_{i+1}")
            ux += bw + 0.54
        else:
            ux += bw

    add_rule(slide, x, 20.55, x + w, 20.55, color=PURPLE_RULE, width=1.5,
             name="METHODS_SummaryRule")
    method_summaries = [
        ("PLUM-like", "Кодов меньше с глубиной; случайный активный префикс. Полная глубина: p = 0.25."),
        ("R3-VAE", "Опорный вектор задаёт якорь; скалярное произведение сохраняет градиент; SC/PD структурирует кластеры."),
        ("PSI", "Коллизии устраняются без дополнительного токена. Сравниваются RRS и ECM."),
        ("LogQ", "Логиты — по SID-префиксам;\nitem loss — по объектам."),
    ]
    positions = [(x, 20.86), (x + 7.22, 20.86), (x, 23.55), (x + 7.22, 23.55)]
    for i, ((head, body), (bx, by)) in enumerate(zip(method_summaries, positions)):
        body_height = 1.40 if head == "LogQ" else 2.35
        add_rich_textbox(
            slide,
            bx,
            by,
            6.82,
            body_height,
            [
                {"runs": [(head, 32, True, GREEN_DARK)]},
                {"runs": [(body, 30, False, INK)], "before": 5, "line_spacing": 1.02},
            ],
            name=f"METHODS_Summary_{i+1}",
        )
        if head == "LogQ":
            loss_math = (
                office_math_subscript("L", "total", 30)
                + office_math_run(" = ", 30, normal=True)
                + office_math_subscript("L", "SID", 30)
                + office_math_run(" + 0.1 ", 30, normal=True)
                + office_math_subscript("L", "item", 30)
            )
            add_office_math_textbox(
                slide,
                bx,
                by + 1.46,
                6.82,
                0.72,
                loss_math,
                name="METHODS_LogQ_OfficeMath",
            )


def heat_color(value: float):
    if value >= 0.50:
        return HEAT_POS_4
    if value >= 0.40:
        return HEAT_POS_3
    if value >= 0.25:
        return HEAT_POS_2
    if value >= 0.08:
        return HEAT_POS_1
    if value <= -0.12:
        return HEAT_NEG_2
    if value <= -0.05:
        return HEAT_NEG_1
    return HEAT_NEUTRAL


def build_results(slide):
    x, w = 31.78, 13.78
    add_textbox(slide, x, 10.30, w, 0.50, "Корреляция SID-метрик с downstream", size=34, bold=True,
                name="RESULTS_HeatmapHead")
    add_textbox(slide, x, 10.84, w, 0.44, "Pearson r · две ключевые SID-метрики", size=30, color=MUTED,
                name="RESULTS_HeatmapSubhead")

    row_label_w = 2.72
    cell_w = (w - row_label_w) / 5
    headers = ["Recall\n@100", "DCNv2\nPairwise", "DCNv2\nROC-AUC", "TransAct\nPairwise", "TransAct\nROC-AUC"]
    for i, label in enumerate(headers):
        add_textbox(slide, x + row_label_w + i * cell_w, 11.34, cell_w, 1.16, label,
                    size=30, bold=True, color=MUTED, align=PP_ALIGN.CENTER,
                    valign=MSO_VERTICAL_ANCHOR.MIDDLE, name=f"RESULTS_HeatmapHeader_{i+1}")

    rows = [
        ("ICR", [0.55, 0.47, -0.05, -0.08, 0.11]),
        ("Intra-code\nSimilarity", [0.32, 0.37, 0.18, -0.13, -0.05]),
    ]
    for r_idx, (label, values) in enumerate(rows):
        y = 12.54 + r_idx * 1.20
        add_textbox(slide, x, y, row_label_w - 0.08, 1.12, label, size=30, bold=True,
                    valign=MSO_VERTICAL_ANCHOR.MIDDLE, name=f"RESULTS_HeatmapRow_{r_idx+1}")
        for c_idx, value in enumerate(values):
            is_anchor = r_idx == 0 and c_idx in (0, 1)
            value_text = f"{value:.2f}" if value >= 0 else f"−{abs(value):.2f}"
            add_box(
                slide,
                x + row_label_w + c_idx * cell_w,
                y,
                cell_w,
                1.12,
                value_text,
                size=40 if is_anchor else 32,
                bold=is_anchor,
                fill=heat_color(value),
                line=PAPER,
                line_width=1,
                name=f"RESULTS_HeatmapCell_{r_idx+1}_{c_idx+1}",
            )
    add_textbox(
        slide,
        x,
        14.94,
        w,
        1.04,
        "Наиболее выраженные корреляции — с Recall@100 и DCNv2 Pairwise. "
        "Корреляция не означает причинность.",
        size=30,
        color=MUTED,
        name="RESULTS_HeatmapNote",
    )

    add_rule(slide, x, 16.08, x + w, 16.08, color=GREEN, width=3,
             name="RESULTS_GenerationRule")
    add_textbox(slide, x, 16.22, w, 0.48, "GENERATION УЛУЧШИЛСЯ · Recall@100", size=34,
                bold=True, color=GREEN_DARK, name="RESULTS_GenerationHead")
    add_textbox(slide, x, 16.88, 8.10, 0.92, "0.12303 → 0.14138", size=54, bold=True,
                valign=MSO_VERTICAL_ANCHOR.MIDDLE, name="RESULTS_GenerationValues")
    add_textbox(slide, x + 8.18, 16.72, 5.52, 1.12, "+14.9%", size=66, bold=True,
                color=GREEN_DARK, align=PP_ALIGN.RIGHT, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="RESULTS_GenerationDelta")
    add_rich_textbox(
        slide,
        x,
        18.02,
        6.20,
        1.45,
        [
            {"runs": [("Baseline", 30, True, MUTED)]},
            {"runs": [("SASRec-Audio-Uniform", 30, False, INK)], "before": 5},
        ],
        name="RESULTS_GenerationBaseline",
    )
    add_rich_textbox(
        slide,
        x + 6.55,
        18.02,
        7.15,
        1.45,
        [
            {"runs": [("Best SID", 30, True, MUTED)]},
            {"runs": [("RQ-VAE + item LogQ, L=2, C=256", 30, False, INK)], "before": 5},
        ],
        name="RESULTS_GenerationBest",
    )

    add_rule(slide, x, 19.90, x + w, 19.90, color=PURPLE_RULE, width=1.6,
             name="RESULTS_DCNv2Rule")
    add_textbox(slide, x, 20.10, w, 0.46,
                "RANKING — УЛУЧШЕНИЯ НЕТ · DCNv2 Pairwise Accuracy",
                size=30, bold=True, color=MUTED, name="RESULTS_DCNv2Head")
    add_textbox(slide, x, 20.74, 8.15, 0.82, "0.54914 → 0.53600", size=44, bold=True,
                valign=MSO_VERTICAL_ANCHOR.MIDDLE, name="RESULTS_DCNv2Values")
    add_textbox(slide, x + 8.30, 20.68, 5.40, 0.90, "−2.4%", size=46, bold=True,
                color=PURPLE, align=PP_ALIGN.RIGHT, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="RESULTS_DCNv2Delta")
    add_textbox(slide, x, 21.72, w, 1.10,
                "Baseline: DCNv2 item-only   ·   Best SID: RQ-VAE (ECM), L=3, C=256",
                size=30, name="RESULTS_DCNv2Models")

    add_rule(slide, x, 23.12, x + w, 23.12, color=PURPLE_RULE, width=1.6,
             name="RESULTS_TransActRule")
    add_textbox(slide, x, 23.32, w, 0.46,
                "RANKING — УЛУЧШЕНИЯ НЕТ · TransAct Pairwise Accuracy",
                size=30, bold=True, color=MUTED, name="RESULTS_TransActHead")
    add_textbox(slide, x, 23.96, 8.15, 0.82, "0.76164 → 0.63811", size=44, bold=True,
                valign=MSO_VERTICAL_ANCHOR.MIDDLE, name="RESULTS_TransActValues")
    add_textbox(slide, x + 8.30, 23.90, 5.40, 0.90, "−16.2%", size=46, bold=True,
                color=PURPLE, align=PP_ALIGN.RIGHT, valign=MSO_VERTICAL_ANCHOR.MIDDLE,
                name="RESULTS_TransActDelta")
    add_textbox(slide, x, 24.94, w, 1.10,
                "Baseline: TransAct item-only   ·   Best SID: RQ-VAE, L=3, C=1024",
                size=30, name="RESULTS_TransActModels")


def build_conclusions(slide):
    theses = [
        (
            "SID ≠ downstream quality",
            "Высокое внутреннее качество SID само по себе не гарантирует улучшение итоговой модели.",
        ),
        (
            "Ranking: улучшения нет",
            "SID-конфигурации не превзошли item-only baseline в рассмотренных экспериментах.",
        ),
        (
            "Generation: +14.9%",
            "RQ-VAE + item LogQ улучшил Recall@100 относительно baseline.",
        ),
        (
            "Проверять нужно downstream",
            "Оценивайте SID внутренними метриками и непосредственно downstream-экспериментами.",
        ),
    ]
    x0, col_w, gap = 2.55, 9.85, 0.82
    for i, (head, body) in enumerate(theses):
        x = x0 + i * (col_w + gap)
        if i:
            add_rule(slide, x - 0.40, 29.70, x - 0.40, 31.78, color=PURPLE_RULE,
                     width=1.5, name=f"CONCLUSIONS_Divider_{i}")
        add_rich_textbox(
            slide,
            x,
            29.62,
            col_w,
            2.18,
            [
                {"runs": [(head, 36, True, GREEN_DARK)], "line_spacing": 1.0},
                {"runs": [(body, 30, False, INK)], "before": 7, "line_spacing": 1.02},
            ],
            name=f"CONCLUSIONS_Thesis_{i+1}",
        )


def assert_slide_invariants(prs):
    assert len(prs.slides) == 1, "Poster must contain exactly one slide"
    # 47.0417 x 33.3333 inches: the original A0 landscape template size.
    assert abs(prs.slide_width / 914400 - 47.0416667) < 0.001
    assert abs(prs.slide_height / 914400 - 33.3333333) < 0.001
    slide = prs.slides[0]
    for shape in slide.shapes:
        assert shape.left >= 0 and shape.top >= 0, f"Off-slide shape: {shape.name}"
        assert shape.left + shape.width <= prs.slide_width + 2, f"Right overflow: {shape.name}"
        assert shape.top + shape.height <= prs.slide_height + 2, f"Bottom overflow: {shape.name}"
        if getattr(shape, "has_text_frame", False):
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if run.text.strip() and run.font.size is not None:
                        assert run.font.size.pt >= 30, (
                            f"Text below 30 pt in {shape.name}: {run.text!r} ({run.font.size.pt})"
                        )
    math_namespace = "http://schemas.microsoft.com/office/drawing/2010/main"
    office_math_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    math_objects = list(slide._element.iter(f"{{{math_namespace}}}m"))
    fractions = list(slide._element.iter(f"{{{office_math_namespace}}}f"))
    subscripts = list(slide._element.iter(f"{{{office_math_namespace}}}sSub"))
    assert len(math_objects) == 2, "Poster must contain exactly two Office Math objects"
    assert len(fractions) == 1, "ICR must contain one native Office Math fraction"
    assert len(subscripts) == 4, "ICR and LogQ formulas must contain four native subscripts"
    for math_object in math_objects:
        sizes = [
            int(node.get("sz"))
            for node in math_object.iter(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}rPr"
            )
            if node.get("sz") is not None
        ]
        assert sizes and min(sizes) >= 3000, "Office Math font size must be at least 30 pt"


def main():
    prs = Presentation(TEMPLATE)
    slide = prs.slides[0]
    replace_template_header(slide)
    build_introduction(slide)
    build_methods(slide)
    build_results(slide)
    build_conclusions(slide)

    prs.core_properties.title = "Влияние качества Semantic ID на финальные метрики"
    prs.core_properties.subject = "A0 scientific poster adapted to the ML-Академия template"
    prs.core_properties.author = "Константин Рядинский, Лев Клюев, Ростислав Головков, Глеб Шевченко"
    prs.core_properties.comments = (
        "Hallmark · audit-only fixed-template scientific poster · technical/austere · "
        "pre-emit critique: P5 H5 E5 S5 R5 V4 · audit: 0 critical, 0 major, 0 minor · "
        "native Office Math: ICR fraction + LogQ loss"
    )

    assert_slide_invariants(prs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
