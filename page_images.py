"""
매뉴얼 PDF 페이지 → 이미지(PNG) 변환
- 답변의 근거 페이지를 그림 그대로 보여주기 위해 사용 (메뉴 화면, 디스플레이 그림 포함)
- 한 번 만든 이미지는 data/page_images/<manual_id>/<페이지>.png 에 저장해 두고 재사용
"""
import csv
from pathlib import Path

BASE = Path(__file__).parent
CATALOG = BASE / "data" / "manual_catalog.csv"
OUT_DIR = BASE / "data" / "page_images"
SCALE = 1.6  # 이미지 확대 배율 (클수록 선명하지만 파일이 커짐)


def _pdf_path(manual_id: str) -> Path:
    with open(CATALOG, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["manual_id"] == manual_id:
                return BASE / row["pdf_path"]
    raise FileNotFoundError(f"catalog에 없는 매뉴얼: {manual_id}")


def page_image(manual_id: str, page: int) -> Path:
    """manual_id 매뉴얼의 page(PDF 페이지 번호, 1부터)를 PNG로 만들어 경로 반환"""
    out = OUT_DIR / manual_id / f"{page}.png"
    if out.exists():
        return out

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(_pdf_path(manual_id)))
    try:
        if not 1 <= page <= len(pdf):
            raise ValueError(f"페이지 범위 초과: {page} (총 {len(pdf)}페이지)")
        img = pdf[page - 1].render(scale=SCALE).to_pil()
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
    finally:
        pdf.close()
    return out
