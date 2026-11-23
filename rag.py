"""
CAS 기술지원 챗봇 - 핵심 로직 (RAG: 검색 + 답변 생성)

흐름 (수업 10주차: "조회·계산은 Python Tool, 판단·설명은 LLM")
  ① select_manual()  : 제품군·국가로 사용할 매뉴얼 결정        → Python 규칙
  ② search()         : 매뉴얼에서 질문과 관련된 페이지 검색     → Python (임베딩 + 키워드)
  ③ generate_answer(): 검색된 페이지만 근거로 답변 생성         → Gemini
  ④ answer()         : ①~③을 묶어서 최종 결과(답변/근거/이관여부) 반환
"""
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

BASE = Path(__file__).parent
CATALOG = BASE / "data" / "manual_catalog.csv"
INDEX_DIR = BASE / "data" / "index"

load_dotenv(BASE / ".env")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")
TOP_K = 4  # Gemini에게 넘겨줄 페이지 수

KOREA_NAMES = {"south korea", "korea", "kr", "대한민국", "한국", "republic of korea"}

from gemini_client import generate, get_client  # 재시도 + 대체 모델 기능 포함
from figures import load_figures                 # 매뉴얼 그림(화면/사진) 위치 정보

FIG_MARK = re.compile(r"\{\{그림:(\d+-\d+)\}\}")  # 답변 속 그림 표시 형식: {{그림:30-5}}


# ─────────────────────────────────────────────
# ① 매뉴얼 선택 (Python 규칙)
# ─────────────────────────────────────────────
def country_to_region(country: str) -> str:
    return "KR" if (country or "").strip().lower() in KOREA_NAMES else "UN"


def select_manual(product: str, country: str, model: str = ""):
    """제품군 + 모델 + 국가(국내 KOR / 해외 ENG)에 맞는 매뉴얼을 catalog에서 찾음. 없으면 None
    - catalog 의 models 칸: 그 매뉴얼이 적용되는 모델명 (세미콜론 ; 으로 구분)
    - models 칸이 비어 있으면 해당 제품군의 '공통 매뉴얼'로 취급
    """
    region = country_to_region(country)
    model_key = (model or "").strip().upper()
    common = None
    with open(CATALOG, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["product"].lower() != (product or "").lower() or row["region"] != region:
                continue
            models = [m.strip().upper() for m in row["models"].split(";") if m.strip()]
            if not models:
                common = common or row          # 제품군 공통 매뉴얼 (모델 지정 없음)
            elif model_key in models:
                return row                      # 모델 전용 매뉴얼 우선
    return common


# ─────────────────────────────────────────────
# ② 매뉴얼 검색 (임베딩 유사도 + 키워드 일치)
# ─────────────────────────────────────────────
_index_cache = {}


def load_index(manual_id: str):
    if manual_id not in _index_cache:
        pages = json.loads((INDEX_DIR / f"{manual_id}.json").read_text(encoding="utf-8"))
        npy = INDEX_DIR / f"{manual_id}.npy"
        vecs = np.load(npy) if npy.exists() else None
        if vecs is not None and len(vecs) != len(pages):
            print(f"⚠ 임베딩 개수({len(vecs)})가 페이지 수({len(pages)})와 달라 키워드 검색만 사용합니다. "
                  f"step2_build_index.py 를 다시 실행하세요.")
            vecs = None
        if vecs is not None:
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        _index_cache[manual_id] = (pages, vecs)
    return _index_cache[manual_id]


def keyword_features(text: str):
    """키워드 비교용 특징: 영문/숫자 단어 + 한글은 2글자씩 쪼갠 조각
    (예: '영점이' → '영점','점이')  → 조사가 붙어도 매칭되도록"""
    text = text.lower()
    feats = set(re.findall(r"[a-z0-9/]{2,}", text))
    for word in re.findall(r"[가-힣]+", text):
        feats.update(word[i:i + 2] for i in range(len(word) - 1))
    return feats


def keyword_score(query: str, page_text: str) -> float:
    q = keyword_features(query)
    if not q:
        return 0.0
    page_norm = re.sub(r"\s+", " ", page_text.lower())
    score = sum(1 for f in q if f in page_norm) / len(q)
    # 에러 코드처럼 영문 대문자 구절이 그대로 있으면 가산점 (예: "ADM NOT IN CODE")
    for phrase in re.findall(r"[A-Za-z][A-Za-z /\-]{4,}[A-Za-z]", query):
        if phrase.lower().strip() in page_norm:
            score += 0.5
    return score


def embed_query(query: str):
    res = get_client().models.embed_content(model=EMBED_MODEL, contents=[query])
    v = np.array(res.embeddings[0].values, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def search(manual_id: str, query: str, top_k: int = TOP_K):
    """질문과 관련성이 높은 페이지 top_k개 반환"""
    pages, vecs = load_index(manual_id)

    kw = np.array([keyword_score(query, p["text"]) for p in pages])
    if kw.max() > 0:
        kw = kw / kw.max()

    if vecs is not None:
        try:
            sem = vecs @ embed_query(query)
            sem = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
            score = 0.6 * sem + 0.4 * kw
        except Exception as e:
            print(f"⚠ 임베딩 검색 실패 → 키워드 검색만 사용: {e}")
            score = kw
    else:
        score = kw

    # 목차 페이지, 빈 페이지는 제외
    for i, p in enumerate(pages):
        if p["is_toc"] or len(p["text"]) < 30:
            score[i] = -1

    order = np.argsort(-score)[:top_k]
    return [dict(pages[i], score=round(float(score[i]), 3)) for i in order]


# ─────────────────────────────────────────────
# ③ Gemini 답변 생성
# ─────────────────────────────────────────────
class GeminiAnswer(BaseModel):
    answer: str            # 사용자에게 보여줄 답변 (마크다운)
    found: bool            # 매뉴얼에서 해결 방법을 찾았는지
    escalate: bool         # 기술문의(연구소/AS) 접수가 필요한지
    escalate_reason: str   # 접수가 필요한 이유 (없으면 빈 문자열)
    used_pages: list[int]  # 답변 근거로 사용한 PDF 페이지 번호


SYSTEM_PROMPT = """너는 CAS 상업용 전자저울 기술지원 챗봇이다.
규칙:
1. 반드시 [매뉴얼 발췌] 안의 내용만 근거로 답한다. 매뉴얼에 없는 메뉴코드, 수치, 절차를 지어내지 않는다.
2. 답변은 한국어로, 사용자가 바로 따라 할 수 있게 번호 단계로 짧게 쓴다. 메뉴코드(예: MENU 1732)는 그대로 표기한다.
3. 매뉴얼에서 해결 방법을 찾지 못하면 found=false, escalate=true 로 하고 "매뉴얼에서 확인되지 않아 기술문의 접수가 필요합니다"라고 안내한다.
4. 매뉴얼이 'A/S 센터로 문의' 또는 수리가 필요하다고 안내하는 경우, 그 의미를 설명하고 escalate=true 로 한다.
5. used_pages 에는 실제로 근거로 쓴 발췌의 PDF 페이지 번호만 넣는다.
6. 저울의 키(버튼)는 매뉴얼 표기처럼 대괄호로 표시한다. 예: '저장키를 누른다' → [저장] 키, '이송키' → [이송] 키, [취소], [실행], [발행], [영점], [용기]
7. 메뉴 진입 경로는 'MENU 1732 (라벨/티켓 크기)'처럼 메뉴코드와 이름을 함께 쓴다.
8. 각 발췌 아래 [그림 목록]이 있다. 사용자가 따라 할 때 화면/그림을 보면 도움이 되는 단계가 있으면,
   그 단계 문장 바로 다음 줄에 {{그림:ID}} 형식으로 넣는다. (예: {{그림:30-5}})
   - 답변의 각 단계마다, 그 단계와 '주변 설명'이 맞는 그림이 있으면 반드시 그 단계 바로 아래에 넣는다.
     (챗봇 답변은 단계별 설명 + 그림으로 보여주는 것이 원칙)
   - 목록에 있는 ID만 사용한다. 같은 그림을 두 번 넣지 않는다.
"""


def generate_answer(question: str, ctx: dict, pages: list):
    from google.genai import types

    figs = load_figures(ctx["manual_id"])

    def fig_list(page_no):
        items = figs.get(str(page_no), [])
        if not items:
            return "[그림 목록] 없음"
        return "[그림 목록]\n" + "\n".join(
            f"- {f['id']}: 위에서 {i}번째 그림 / 주변 설명: {f['context'] or '(없음)'}"
            for i, f in enumerate(items, 1))

    excerpts = "\n\n".join(
        f"--- [PDF {p['page']}페이지 / 매뉴얼 표기 {p['printed_page']}쪽] ---\n{p['text'][:3500]}\n{fig_list(p['page'])}"
        for p in pages
    )
    prompt = f"""[문의 제품 정보]
제품군: {ctx.get('product')} / 모델: {ctx.get('model')} / 국가: {ctx.get('country')}
펌웨어: {ctx.get('firmware')} / 리비전: {ctx.get('revision')} / 문의자 권한: {ctx.get('role')}

[문의 내용]
{question}

[매뉴얼 발췌: {ctx.get('manual_name')}]
{excerpts}
"""
    res, used_model = generate(
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=GeminiAnswer,
        ),
    )
    parsed = res.parsed if res.parsed is not None else GeminiAnswer(**json.loads(res.text))
    return parsed, used_model


STEP_MATCH_MIN = 0.35   # 단계 문장과 그림 주변 설명이 이 비율 이상 겹치면 같은 내용으로 판단
MAX_FIGURES = 8


def attach_step_figures(answer_text: str, used: list, pages: list, manual_id: str):
    """답변을 줄(단계) 단위로 보면서, 그림이 없는 단계에 내용이 맞는 그림을 바로 아래 삽입"""
    figs = load_figures(manual_id)
    candidates = [f for p in pages for f in figs.get(str(p), []) if f["context"]]
    if not candidates:
        return answer_text, used

    used = list(used)
    lines = answer_text.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        step = line.strip()
        if len(step) < 6 or FIG_MARK.search(step) or len(used) >= MAX_FIGURES:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if FIG_MARK.search(nxt):
            continue                        # 이미 이 단계 아래에 그림이 있음
        scored = [(keyword_score(step, f["context"]), f) for f in candidates if f["id"] not in used]
        scored = [x for x in scored if x[0] >= STEP_MATCH_MIN]
        if not scored:
            continue
        best = max(x[0] for x in scored)
        # 같은 설명에 그림이 2개(도면 + 사진)인 경우도 함께 표시
        for sc, f in sorted(scored, key=lambda x: -x[0]):
            if sc >= best - 0.05 and len(used) < MAX_FIGURES:
                out.append("{{그림:%s}}" % f["id"])
                used.append(f["id"])
    return "\n".join(out), used


# ─────────────────────────────────────────────
# ④ 전체 처리
# ─────────────────────────────────────────────
def answer(question: str, product: str, model: str = "", country: str = "",
           firmware: str = "", revision: str = "", role: str = "User") -> dict:
    t0 = time.time()
    ctx = dict(product=product, model=model, country=country,
               firmware=firmware, revision=revision, role=role)

    manual = select_manual(product, country, model)
    if manual is None:
        # 매뉴얼이 없는 조합 → Gemini 호출 없이 바로 기술문의 접수 안내
        region = "국내(KOR)" if country_to_region(country) == "KR" else "해외(ENG)"
        return {
            "answer": f"현재 **{product} {model} / {region}** 매뉴얼이 등록되어 있지 않아 자동 안내가 어렵습니다.\n"
                      f"기술문의를 접수해 주시면 담당자가 확인 후 답변드리겠습니다.",
            "found": False,
            "escalate": True,
            "escalate_reason": "해당 모델/지역의 매뉴얼 미등록",
            "manual": None,
            "figures": [],
            "sources": [],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    ctx["manual_name"] = manual["manual_name"]
    ctx["manual_id"] = manual["manual_id"]
    hits = search(manual["manual_id"], question)
    result, used_model = generate_answer(question, ctx, hits)

    # 답변에 들어간 그림 표시 중 실제로 존재하는 그림만 남김 (없는 ID는 제거)
    all_figs = {f["id"]: f for v in load_figures(manual["manual_id"]).values() for f in v}
    allowed_pages = {h["page"] for h in hits}
    answer_text = result.answer
    figures_used = []
    for fid in FIG_MARK.findall(answer_text):
        if fid in all_figs and int(fid.split("-")[0]) in allowed_pages and fid not in figures_used:
            figures_used.append(fid)
        elif fid not in figures_used:
            answer_text = answer_text.replace("{{그림:%s}}" % fid, "")

    # 각 단계(줄)마다 그림이 빠져 있으면 → 근거 페이지 그림 중 그 단계와 맞는 그림을 바로 아래에 자동 삽입
    if result.found:
        answer_text, figures_used = attach_step_figures(
            answer_text, figures_used, result.used_pages or [h["page"] for h in hits], manual["manual_id"])

    answer_text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", answer_text).strip()

    used = set(result.used_pages) or {h["page"] for h in hits}
    sources = [
        {"page": h["page"], "printed_page": h["printed_page"], "title": h["title"],
         "score": h["score"], "snippet": h["text"][:200]}
        for h in hits if h["page"] in used
    ]
    return {
        "answer": answer_text,
        "figures": [{"id": fid, "page": int(fid.split("-")[0]),
                     "context": all_figs[fid]["context"]} for fid in figures_used],
        "found": result.found,
        "escalate": result.escalate,
        "escalate_reason": result.escalate_reason,
        "manual": manual["manual_name"],
        "manual_id": manual["manual_id"],
        "sources": sources,
        "retrieved_pages": [h["page"] for h in hits],
        "model_used": used_model,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
