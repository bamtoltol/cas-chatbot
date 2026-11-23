"""
[2단계] 매뉴얼 PDF → 검색용 인덱스 만들기 (처음 한 번만 실행)

하는 일
  1) data/manual_catalog.csv 에 등록된 매뉴얼 PDF를 읽습니다.
  2) 페이지 단위로 텍스트를 뽑습니다. (1페이지 = 검색 단위 1개)
  3) 각 페이지를 Gemini 임베딩(의미 벡터)으로 바꿉니다.
  4) data/index/<manual_id>.json (텍스트) + .npy (벡터) 로 저장합니다.

실행
  python step2_build_index.py              → 텍스트 + 임베딩 모두 생성
  python step2_build_index.py --no-embed   → 텍스트만 생성 (키워드 검색만 사용)

매뉴얼을 새로 추가하면: manual_catalog.csv 에 한 줄 추가 → 이 파일 다시 실행
"""
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader

BASE = Path(__file__).parent
CATALOG = BASE / "data" / "manual_catalog.csv"
INDEX_DIR = BASE / "data" / "index"

load_dotenv(BASE / ".env")


def extract_pages(pdf_path: Path):
    """PDF를 페이지별 텍스트 목록으로 변환"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)          # 연속 공백 정리
        text = re.sub(r"\n\s*\n+", "\n", text).strip()  # 빈 줄 정리

        # 매뉴얼에 인쇄된 쪽 번호 찾기 (페이지 맨 앞/맨 뒤의 숫자만 있는 줄)
        printed = None
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for cand in (lines[:2] + lines[-2:]) if lines else []:
            if re.fullmatch(r"\d{1,3}", cand):
                printed = int(cand)
                break

        # 목차 페이지 표시 (점선 ..... 이 많은 페이지)
        is_toc = text.count("....") >= 8

        # 페이지 제목: 숫자만 있는 줄을 건너뛴 첫 줄
        title = next((l for l in lines if not re.fullmatch(r"\d{1,3}", l)), "")[:60]

        pages.append({
            "page": i + 1,          # PDF 상의 페이지 번호
            "printed_page": printed, # 매뉴얼에 인쇄된 쪽 번호
            "title": title,
            "is_toc": is_toc,
            "text": text,
        })
    return pages


def embed_texts(texts, model):
    """Gemini 임베딩 API로 텍스트 목록을 벡터로 변환 (20개씩 나눠서 요청)"""
    from gemini_client import get_client  # 30초 시간 제한이 걸린 클라이언트

    client = get_client()
    vectors = []
    batch_mode = True   # 여러 페이지를 한 번에 요청 (모델이 지원하지 않으면 1페이지씩으로 전환)
    batch = 20
    start = 0
    while start < len(texts):
        size = batch if batch_mode else 1
        chunk = [t[:6000] if t.strip() else "(빈 페이지)" for t in texts[start:start + size]]
        for attempt in range(5):
            try:
                res = client.models.embed_content(model=model, contents=chunk)
                break
            except Exception as e:  # 서버 혼잡(503), 사용량 초과(429) 등 → 잠깐 쉬고 재시도
                if getattr(e, "code", None) == 404:
                    raise SystemExit(f"❌ 임베딩 모델 '{model}' 을 찾을 수 없습니다. .env 의 EMBED_MODEL 을 확인하세요.")
                wait = 10 * (attempt + 1)
                print(f"  ⚠ 임베딩 오류: {e}\n  → {wait}초 후 재시도")
                time.sleep(wait)
        else:
            raise SystemExit("❌ 임베딩을 5번 시도했지만 실패했습니다.")

        # 요청한 페이지 수만큼 벡터가 왔는지 검증
        if len(res.embeddings) != len(chunk):
            print(f"  ℹ 이 모델은 여러 페이지를 하나로 합쳐 버립니다 "
                  f"({len(chunk)}개 요청 → {len(res.embeddings)}개 응답) → 1페이지씩 요청으로 전환")
            batch_mode = False
            continue  # 같은 위치부터 1페이지씩 다시

        vectors.extend([e.values for e in res.embeddings])
        start += len(chunk)
        if start % 20 == 0 or start == len(texts):
            print(f"  임베딩 {start}/{len(texts)} 페이지 완료")
        time.sleep(0.3 if not batch_mode else 1)

    assert len(vectors) == len(texts), "임베딩 개수가 페이지 수와 다릅니다."
    return np.array(vectors, dtype=np.float32)


def main():
    use_embed = "--no-embed" not in sys.argv
    embed_model = os.getenv("EMBED_MODEL", "gemini-embedding-2")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(CATALOG, encoding="utf-8-sig") as f:
        manuals = list(csv.DictReader(f))

    for m in manuals:
        pdf_path = BASE / m["pdf_path"]
        if not pdf_path.exists():
            print(f"❌ PDF 없음: {pdf_path}  → manuals 폴더에 파일을 넣어주세요.")
            continue

        print(f"\n📘 {m['manual_name']} 읽는 중...")
        pages = extract_pages(pdf_path)
        empty = sum(1 for p in pages if len(p["text"]) < 30)
        print(f"  총 {len(pages)} 페이지 (텍스트 거의 없는 페이지 {empty}개)")

        out_json = INDEX_DIR / f"{m['manual_id']}.json"
        out_json.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  ✅ 텍스트 저장: {out_json.relative_to(BASE)}")

        if use_embed:
            print(f"  임베딩 생성 중 (모델: {embed_model}) ...")
            vecs = embed_texts([p["text"] for p in pages], embed_model)
            out_npy = INDEX_DIR / f"{m['manual_id']}.npy"
            np.save(out_npy, vecs)
            print(f"  ✅ 임베딩 저장: {out_npy.relative_to(BASE)}  shape={vecs.shape}")

    print("\n완료! 이제 python step3_ask.py 로 질문해보세요.")


if __name__ == "__main__":
    main()
