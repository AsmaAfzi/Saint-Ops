#!/usr/bin/env python3
"""Extract figure/table order from FYP docx."""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

DOCX = r"c:\Users\asmaa\Starred\UNI\Final Proj Docs\Final Year Project Report _Asma Mohammad Afzal.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with zipfile.ZipFile(DOCX) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
    rid_map = {
        rel.get("Id"): rel.get("Target")
        for rel in rels
        if rel.get("Target") and "media/" in rel.get("Target", "")
    }

    body = root.find(f".//{W}body")
    items: list[tuple[str, object, list[str]]] = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            parts = [t.text for t in child.iter(f"{W}t") if t.text]
            text = "".join(parts).strip()
            embeds = [
                b.get(R)
                for b in child.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
            ]
            embeds = [e for e in embeds if e]
            if text or embeds:
                items.append(("p", text, embeds))
        elif tag == "tbl":
            rows = []
            for tr in child.findall(f"{W}tr"):
                cells = []
                for tc in tr.findall(f"{W}tc"):
                    cparts = [t.text for t in tc.iter(f"{W}t") if t.text]
                    cells.append("".join(cparts).strip())
                if any(cells):
                    rows.append(cells)
            items.append(("tbl", rows, []))

    img_num = 0
    tbl_num = 0
    for kind, data, embeds in items:
        if kind == "p" and embeds:
            img_num += 1
            files = [rid_map.get(e, "?") for e in embeds]
            print(f"--- IMAGE {img_num}: {files}")
            if data:
                print(f"  TEXT: {data[:250]}")
        if kind == "p" and data and re.match(r"(?i)^(Fig\.|Figure|Table)", data):
            print(f"CAPTION: {data}")
        if kind == "tbl":
            tbl_num += 1
            rows = data
            header = " | ".join(rows[0][:8]) if rows else ""
            if tbl_num <= 20:
                print(f"TABLE {tbl_num}: {header[:200]}")
                if len(rows) > 1:
                    print(f"  ... ({len(rows)} rows)")


if __name__ == "__main__":
    main()
