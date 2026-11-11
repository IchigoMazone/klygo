"""Render every report page and record extraction/layout checks for visual review."""
import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
QA = ROOT / "tmp/pdfs/klygo"


if __name__ == "__main__":
    QA.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(REPORT / "main.pdf")
    records = []
    for index, page in enumerate(document):
        text = page.get_text()
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        suspicious = [span for span in spans if span["bbox"][0] < 20 or span["bbox"][2] > page.rect.width - 20
                      or span["bbox"][1] < 10 or span["bbox"][3] > page.rect.height - 10]
        record = {"physical_page": index + 1, "characters": len(text), "replacement_character": "\ufffd" in text,
                  "unresolved_reference": "??" in text, "out_of_page_text": suspicious,
                  "start": text[:130].replace("\n", " | ")}
        records.append(record)
        page.get_pixmap(matrix=pymupdf.Matrix(0.85, 0.85), alpha=False).save(QA / f"page-{index+1:03d}.png")
        if index == 0 or index == len(document)-1 or "So sánh kết quả trên ảnh COCO ID" in text or "SHA-256 của ba checkpoint" in text or "Các nhóm trách nhiệm" in text:
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(QA / f"detail-{index+1:03d}.png")
    for start in range(0, len(document), 20):
        sheet = Image.new("RGB", (1000, 5 * 385), "#d6d9de")
        draw = ImageDraw.Draw(sheet)
        for offset in range(min(20, len(document) - start)):
            with Image.open(QA / f"page-{start+offset+1:03d}.png") as image:
                thumb = ImageOps.contain(image, (234, 350))
            x, y = (offset % 4) * 250 + 8, (offset // 4) * 385 + 20
            sheet.paste(thumb, (x, y))
            draw.text((x, y - 15), f"PDF page {start+offset+1}", fill="black")
        sheet.save(QA / f"contact-{start//20+1:02d}.png")
    result = {"page_count": len(document), "pages": records,
              "toc": document.get_toc(), "metadata": document.metadata}
    (QA / "inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(document), "flags": [r for r in records if r["replacement_character"] or r["unresolved_reference"] or r["out_of_page_text"]],
                      "toc": document.get_toc()}, ensure_ascii=False, indent=2))
