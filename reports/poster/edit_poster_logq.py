#!/usr/bin/env python3
"""Apply the narrowly scoped LogQ and Recall@100-vs-ICR poster edits."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"p": P_NS, "a": A_NS}
EMU_PER_INCH = 914400


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def inches(value: float) -> str:
    return str(round(value * EMU_PER_INCH))


def find_shape(root: etree._Element, name: str) -> etree._Element:
    matches = root.xpath(
        f".//p:sp[p:nvSpPr/p:cNvPr[@name='{name}']]", namespaces=NS
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one shape named {name!r}, found {len(matches)}")
    return matches[0]


def find_picture(root: etree._Element, name: str) -> etree._Element:
    matches = root.xpath(
        f".//p:pic[p:nvPicPr/p:cNvPr[@name='{name}']]", namespaces=NS
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one picture named {name!r}, found {len(matches)}")
    return matches[0]


def set_name_and_id(shape: etree._Element, name: str, shape_id: int) -> None:
    props = shape.xpath("./p:nvSpPr/p:cNvPr", namespaces=NS)[0]
    props.set("name", name)
    props.set("id", str(shape_id))


def set_bounds(
    shape: etree._Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    xfrm = shape.xpath("./p:spPr/a:xfrm", namespaces=NS)[0]
    off = xfrm.find(qn(A_NS, "off"))
    ext = xfrm.find(qn(A_NS, "ext"))
    off.set("x", inches(x))
    off.set("y", inches(y))
    ext.set("cx", inches(width))
    ext.set("cy", inches(height))


def set_picture_bounds(
    picture: etree._Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    xfrm = picture.xpath("./p:spPr/a:xfrm", namespaces=NS)[0]
    off = xfrm.find(qn(A_NS, "off"))
    ext = xfrm.find(qn(A_NS, "ext"))
    off.set("x", inches(x))
    off.set("y", inches(y))
    ext.set("cx", inches(width))
    ext.set("cy", inches(height))


def set_body_anchor(shape: etree._Element, anchor: str = "t") -> None:
    body_pr = shape.xpath("./p:txBody/a:bodyPr", namespaces=NS)[0]
    body_pr.set("anchor", anchor)


def replace_text(
    shape: etree._Element,
    paragraphs: list[str],
    *,
    font_size: int,
    color: str,
    bold: bool = False,
    line_spacing: int = 105000,
) -> None:
    tx_body = shape.xpath("./p:txBody", namespaces=NS)[0]
    for paragraph in list(tx_body.findall(qn(A_NS, "p"))):
        tx_body.remove(paragraph)

    for text in paragraphs:
        paragraph = etree.SubElement(tx_body, qn(A_NS, "p"))
        p_pr = etree.SubElement(paragraph, qn(A_NS, "pPr"))
        p_pr.set("algn", "l")
        ln_spc = etree.SubElement(p_pr, qn(A_NS, "lnSpc"))
        etree.SubElement(ln_spc, qn(A_NS, "spcPct"), val=str(line_spacing))
        spc_aft = etree.SubElement(p_pr, qn(A_NS, "spcAft"))
        etree.SubElement(spc_aft, qn(A_NS, "spcPts"), val="0")

        run = etree.SubElement(paragraph, qn(A_NS, "r"))
        run_props = etree.SubElement(run, qn(A_NS, "rPr"))
        run_props.set("lang", "ru-RU")
        run_props.set("sz", str(font_size))
        run_props.set("b", "1" if bold else "0")
        solid_fill = etree.SubElement(run_props, qn(A_NS, "solidFill"))
        etree.SubElement(solid_fill, qn(A_NS, "srgbClr"), val=color)
        etree.SubElement(run_props, qn(A_NS, "latin"), typeface="Arial")
        etree.SubElement(run_props, qn(A_NS, "cs"), typeface="Arial")
        text_node = etree.SubElement(run, qn(A_NS, "t"))
        text_node.text = text

        end_props = etree.SubElement(paragraph, qn(A_NS, "endParaRPr"))
        end_props.set("lang", "ru-RU")
        end_props.set("sz", str(font_size))
        end_props.set("b", "1" if bold else "0")
        end_fill = etree.SubElement(end_props, qn(A_NS, "solidFill"))
        etree.SubElement(end_fill, qn(A_NS, "srgbClr"), val=color)
        etree.SubElement(end_props, qn(A_NS, "latin"), typeface="Arial")
        etree.SubElement(end_props, qn(A_NS, "cs"), typeface="Arial")


def replace_text_nodes(shape: etree._Element, values: list[str]) -> None:
    """Replace visible text while preserving all existing run formatting."""
    text_nodes = shape.xpath(".//a:t", namespaces=NS)
    if len(values) > len(text_nodes):
        raise RuntimeError(
            f"Shape has {len(text_nodes)} text nodes, cannot place {len(values)} values"
        )
    for node, value in zip(text_nodes, values):
        node.text = value
    for node in text_nodes[len(values) :]:
        node.text = ""


def edit_slide(slide_xml: bytes) -> bytes:
    root = etree.fromstring(slide_xml)
    shape_tree = root.xpath("./p:cSld/p:spTree", namespaces=NS)[0]

    existing_ids = [
        int(value)
        for value in root.xpath(".//p:cNvPr/@id", namespaces=NS)
        if value.isdigit()
    ]
    next_id = max(existing_ids) + 1

    logq_head = find_shape(root, "METHODS_LogQHead")
    logq_body = find_shape(root, "METHODS_LogQWhat")
    body_template = copy.deepcopy(logq_body)

    # Keep LogQ exclusively about prefix-aware correction for Semantic IDs.
    set_bounds(logq_head, x=22.89, y=21.779, width=7.38, height=0.50)
    replace_text(
        logq_head,
        ["LogQ"],
        font_size=3400,
        color="087C48",
        bold=True,
    )
    set_bounds(logq_body, x=22.89, y=22.27, width=7.38, height=0.86)
    set_body_anchor(logq_body, "t")
    replace_text(
        logq_body,
        [
            "В SID objective корректируем logits:",
            "logit − log q(token | prefix).",
        ],
        font_size=2400,
        color="111116",
        line_spacing=100000,
    )

    # Add item head + item LogQ as a genuinely separate method block.
    item_head = copy.deepcopy(logq_head)
    set_name_and_id(item_head, "METHODS_ItemLogQHead", next_id)
    next_id += 1
    set_bounds(item_head, x=22.89, y=23.15, width=7.38, height=0.50)
    replace_text(
        item_head,
        ["Item retrieval + item LogQ"],
        font_size=3000,
        color="087C48",
        bold=True,
    )

    item_body = body_template
    set_name_and_id(item_body, "METHODS_ItemLogQWhat", next_id)
    set_bounds(item_body, x=22.89, y=23.66, width=7.38, height=1.68)
    set_body_anchor(item_body, "t")
    replace_text(
        item_body,
        [
            "• параллельная item-level ветка выбирает target item",
            "  среди items батча;",
            "• item LogQ корректирует retrieval scores",
            "  с учётом частоты item;",
            "• retrieval loss добавляем к SID loss.",
        ],
        font_size=2000,
        color="111116",
        line_spacing=100000,
    )

    # Center the updated retrieval-loss formula under the item-level block.
    formula = find_picture(root, "Рисунок 29")
    set_picture_bounds(formula, x=23.73, y=25.42, width=5.70, height=0.95)

    # Keep the result summary consistent with the refreshed experiment data.
    replace_text_nodes(find_shape(root, "RESULTS_TableCell_1_0"), ["DIGER"])
    replace_text_nodes(find_shape(root, "RESULTS_TableCell_1_1"), ["0.15098"])
    replace_text_nodes(find_shape(root, "RESULTS_TableCell_1_2"), ["+22.7%"])
    replace_text_nodes(
        find_shape(root, "CONCLUSIONS_Thesis_3"),
        ["Generation: +22.7%", "DIGER дал лучший Recall@100."],
    )

    shape_tree.append(item_head)
    shape_tree.append(item_body)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def retrieval_formula_svg() -> bytes:
    """Return a portable vector equation with real typographic subscripts."""
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="150" viewBox="0 0 900 150">
  <text x="450" y="92" text-anchor="middle" fill="#111116"
        font-family="Times New Roman, Times, serif" font-size="58">
    <tspan font-style="italic">L</tspan><tspan baseline-shift="sub" font-size="34">total</tspan>
    <tspan> = </tspan>
    <tspan font-style="italic">L</tspan><tspan baseline-shift="sub" font-size="34">SID</tspan>
    <tspan> + 0.1 </tspan>
    <tspan font-style="italic">L</tspan><tspan baseline-shift="sub" font-size="34">retrieval</tspan>
  </text>
</svg>
'''


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: edit_poster_logq.py INPUT.pptx OUTPUT.pptx CURRENT_RECALL_ICR.svg"
        )

    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    chart = Path(sys.argv[3]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    replacements = {
        "ppt/slides/slide1.xml": edit_slide,
        "ppt/media/image6.svg": lambda _: chart.read_bytes(),
        "ppt/media/image8.svg": lambda _: retrieval_formula_svg(),
    }

    with ZipFile(source, "r") as input_zip, ZipFile(
        output, "w", compression=ZIP_DEFLATED
    ) as output_zip:
        names = set(input_zip.namelist())
        missing = set(replacements) - names
        if missing:
            raise RuntimeError(f"Missing expected package members: {sorted(missing)}")
        for info in input_zip.infolist():
            data = input_zip.read(info.filename)
            if info.filename in replacements:
                data = replacements[info.filename](data)
            output_zip.writestr(info, data)

    print(output)


if __name__ == "__main__":
    main()
