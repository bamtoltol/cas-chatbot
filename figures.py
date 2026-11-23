"""
매뉴얼 PDF에서 '그림'(디스플레이 화면, 사진, 도면) 영역을 찾아 잘라내는 모듈

원리
  - PDF 안의 사진(이미지 객체)            → 그림
  - 진한 색으로 채워진 사각형(LCD 화면 틀) → 그림
  - 작은 선들이 모여 있는 곳(도면)          → 그림
  - 표(가로줄이 많고 글자가 많음), 페이지 옆 장식 띠 → 제외
  각 그림마다 '주변 설명 글'(그림 왼쪽/위쪽 텍스트)을 함께 저장해서
  Gemini가 어떤 단계에 어떤 그림을 붙일지 고를 수 있게 합니다.
"""
import ctypes
import csv
import json
from pathlib import Path

BASE = Path(__file__).parent
CATALOG = BASE / "data" / "manual_catalog.csv"
INDEX_DIR = BASE / "data" / "index"
CROP_DIR = BASE / "data" / "figure_images"
SCALE = 2.0  # 잘라낸 그림 해상도 배율


def _pdf_path(manual_id):
    with open(CATALOG, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["manual_id"] == manual_id:
                return BASE / row["pdf_path"]
    raise FileNotFoundError(f"catalog에 없는 매뉴얼: {manual_id}")


def _fill_brightness(raw, obj):
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    if raw.FPDFPageObj_GetFillColor(obj.raw, r, g, b, a):
        return (r.value + g.value + b.value) / 3
    return None


def _merge(boxes, gap):
    """가까이 붙어 있는 사각형들을 하나로 합침. box = [l, b, r, t, 개수]"""
    boxes = [list(x) for x in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                c = boxes[i]
                if a[0] - gap <= c[2] and c[0] - gap <= a[2] and a[1] - gap <= c[3] and c[1] - gap <= a[3]:
                    a = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3]), a[4] + c[4]]
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


def _clean(text):
    return " ".join((text or "").split())


def detect_page(page, textpage):
    """한 페이지에서 그림 영역 찾기 → [{'bbox':[l,b,r,t], 'context': '주변 설명'}]"""
    import pypdfium2.raw as raw

    W, H = page.get_size()
    left_band, right_band = W * 0.11, W * 0.89   # 페이지 옆 장식 띠 영역
    strong, small, hlines = [], [], []

    for o in page.get_objects(max_depth=4):
        if o.type not in (raw.FPDF_PAGEOBJ_PATH, raw.FPDF_PAGEOBJ_IMAGE):
            continue
        l, b, r, t = o.get_bounds()
        w, h = r - l, t - b
        if l >= right_band - 4 or r <= left_band + 4:
            continue
        if o.type == raw.FPDF_PAGEOBJ_IMAGE:
            if w > 25 and h > 15:
                strong.append((l, b, r, t, 50))
            continue
        fill = _fill_brightness(raw, o)
        if h < 2.5 and w > 60:
            hlines.append((l, b, r, t))           # 표의 가로줄 후보
        if fill is not None and fill < 140 and w > 60 and h > 20:
            strong.append((l, b, r, t, 50))       # 진한 사각형 = 디스플레이 화면 틀
            continue
        if fill is not None and fill > 200 and w > 60:
            continue                              # 연회색 띠(메뉴 경로, 표 머리글)
        if w < 200 and h < 200:
            small.append((l, b, r, t, 1))         # 도면을 이루는 작은 선들

    clusters = [c for c in _merge(small, 4) if c[4] >= 12 and (c[2] - c[0]) > 40 and (c[3] - c[1]) > 30]
    boxes = [f for f in _merge(strong + clusters, 6) if (f[2] - f[0]) > 40 and (f[3] - f[1]) > 20]

    figures = []
    for l, b, r, t, _ in boxes:
        inside_text = _clean(textpage.get_text_bounded(l, b, r, t))
        long_lines = sum(1 for (hl, hb, hr, ht) in hlines
                         if hl >= l - 2 and hr <= r + 2 and hb >= b - 2 and ht <= t + 2 and (hr - hl) > (r - l) * 0.5)
        if long_lines >= 3 and len(inside_text) > 40:
            continue                              # 표로 판단 → 제외
        if (t - b) > 330:
            continue                              # 페이지 대부분을 차지 → 표/목록으로 판단
        # 주변 설명: 그림 왼쪽 같은 높이의 글 + 그림 바로 위의 글
        left_text = _clean(textpage.get_text_bounded(left_band, b - 4, max(left_band + 1, l - 2), t + 4))
        above_text = _clean(textpage.get_text_bounded(left_band, t, right_band, min(H, t + 28)))
        figures.append({
            "bbox": [round(l, 1), round(b, 1), round(r, 1), round(t, 1)],
            "context": (above_text + " / " + left_text).strip(" /")[:200],
            "inside": inside_text[:80],
        })

    figures.sort(key=lambda f: (-f["bbox"][3], f["bbox"][0]))   # 위→아래, 왼쪽→오른쪽 순서
    return figures


def build_figures(manual_id):
    """매뉴얼 전체 페이지의 그림 목록 생성 → data/index/<id>_figures.json"""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(_pdf_path(manual_id)))
    result = {}
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            figs = detect_page(page, tp)
            for n, f in enumerate(figs, 1):
                f["id"] = f"{i + 1}-{n}"
            if figs:
                result[str(i + 1)] = figs
    finally:
        pdf.close()
    out = INDEX_DIR / f"{manual_id}_figures.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


_cache = {}


def load_figures(manual_id):
    """그림 목록 불러오기 (없으면 처음 한 번 자동 생성, 수 초 소요)"""
    if manual_id not in _cache:
        path = INDEX_DIR / f"{manual_id}_figures.json"
        if path.exists():
            _cache[manual_id] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print("  (처음 한 번) 매뉴얼 그림 위치를 분석하는 중...")
            _cache[manual_id] = build_figures(manual_id)
    return _cache[manual_id]


def figure_image(manual_id, fig_id):
    """그림 ID(예: '94-1')에 해당하는 영역을 잘라 PNG로 저장하고 경로 반환"""
    out = CROP_DIR / manual_id / f"{fig_id}.png"
    if out.exists():
        return out
    page_no = int(fig_id.split("-")[0])
    fig = next((f for f in load_figures(manual_id).get(str(page_no), []) if f["id"] == fig_id), None)
    if fig is None:
        raise ValueError(f"그림 없음: {fig_id}")

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(_pdf_path(manual_id)))
    try:
        page = pdf[page_no - 1]
        W, H = page.get_size()
        img = page.render(scale=SCALE).to_pil()
        l, b, r, t = fig["bbox"]
        pad = 4
        box = (max(0, (l - pad) * SCALE), max(0, (H - t - pad) * SCALE),
               min(W * SCALE, (r + pad) * SCALE), min(H * SCALE, (H - b + pad) * SCALE))
        out.parent.mkdir(parents=True, exist_ok=True)
        img.crop(tuple(int(v) for v in box)).save(out)
    finally:
        pdf.close()
    return out
