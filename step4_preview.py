"""
[미리보기] 챗봇 답변을 '채팅 화면'처럼 브라우저에 보여주기
- 설명 → 해당 그림 → 다음 설명 → 그림 순서로, Lovable 연동 후 모습을 미리 확인
- 실행: python step4_preview.py
- 결과: preview.html 이 만들어지고 브라우저가 자동으로 열립니다. (과제 캡처용으로도 사용)
"""
import base64
import html
import re
import webbrowser
from pathlib import Path

from figures import figure_image
from rag import FIG_MARK, answer

OUT = Path(__file__).parent / "preview.html"
history = []  # (질문, 결과) 누적 → 대화처럼 이어서 표시


def img_tag(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img class="fig" src="data:image/png;base64,{data}">'


def md_inline(text):
    """**굵게**, [키] 강조 정도만 간단히 변환"""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\[([^\[\]]{1,12})\]", r'<span class="key">\1</span>', t)
    return t


def render_answer(r):
    parts = []
    for line in r["answer"].split("\n"):
        m = FIG_MARK.fullmatch(line.strip())
        if m:
            try:
                parts.append(img_tag(figure_image(r["manual_id"], m.group(1))))
            except Exception:
                pass
        elif line.strip():
            parts.append(f"<p>{md_inline(FIG_MARK.sub('', line))}</p>")
    body = "\n".join(parts)
    src = ", ".join(f"매뉴얼 {s['printed_page']}쪽" for s in r["sources"])
    meta = f'<div class="meta">📄 근거: {src or "없음"} · ⏱ {r["elapsed_ms"] / 1000:.1f}초 · 🤖 {r.get("model_used", "-")}</div>'
    esc = (f'<div class="esc">🚩 {html.escape(r["escalate_reason"] or "추가 확인이 필요합니다")}'
           f'<button>기술문의 접수</button></div>') if r["escalate"] else ""
    return body + esc + meta


def write_page(ctx):
    chats = "".join(
        f'<div class="row user"><div class="bubble u">{html.escape(q)}</div></div>'
        f'<div class="row"><div class="bubble a"><div class="who">CAS Scale Assistant</div>{render_answer(r)}</div></div>'
        for q, r in history)
    OUT.write_text(f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>CAS 기술지원 챗봇 미리보기</title><style>
body{{margin:0;font-family:'Malgun Gothic',sans-serif;background:#eef1f5}}
header{{background:#0b4ea2;color:#fff;padding:14px 22px;font-weight:700}}
.info{{padding:8px 22px;font-size:13px;color:#555;background:#fff;border-bottom:1px solid #ddd}}
.wrap{{max-width:760px;margin:0 auto;padding:20px}}
.row{{display:flex;margin:12px 0}} .row.user{{justify-content:flex-end}}
.bubble{{border-radius:12px;padding:14px 18px;max-width:85%;line-height:1.6;box-shadow:0 1px 3px #0002}}
.u{{background:#0b4ea2;color:#fff}} .a{{background:#fff}}
.who{{color:#0b4ea2;font-size:12px;font-weight:700;margin-bottom:6px}}
.a p{{margin:6px 0}} .fig{{display:block;max-width:100%;max-height:260px;margin:6px 0 12px 18px;border:1px solid #ddd;border-radius:6px}}
.key{{background:#eef2f8;border:1px solid #b7c4d8;border-radius:4px;padding:0 5px;font-weight:700}}
.meta{{margin-top:10px;font-size:12px;color:#888}}
.esc{{margin-top:10px;padding:10px;background:#fff4e5;border-radius:8px;color:#a35c00}}
.esc button{{margin-left:10px;background:#0b4ea2;color:#fff;border:0;border-radius:6px;padding:6px 12px}}
</style></head><body><header>CAS Scale Assistant</header>
<div class="info">{html.escape(ctx)}</div><div class="wrap">{chats}</div></body></html>""", encoding="utf-8")


if __name__ == "__main__":
    print("=" * 60)
    print(" CAS 챗봇 채팅화면 미리보기   종료: q")
    print("=" * 60)
    product = input("제품군 [1] Label Scale  [2] Simple Weight Scale  (기본 1): ").strip()
    product = "Simple Weight Scale" if product == "2" else "Label Scale"
    model = input("모델명 (기본 CL5200N): ").strip() or "CL5200N"
    country = input("국가 (기본 South Korea): ").strip() or "South Korea"
    ctx = f"{product} · {model} · {country}"

    opened = False
    while True:
        q = input("\n❓ 문의 내용: ").strip()
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        r = answer(q, product=product, model=model, country=country)
        history.append((q, r))
        write_page(ctx)
        print(f"✅ 답변 완료 → {OUT.name} 갱신 (브라우저에서 새로고침 F5)")
        if not opened:
            webbrowser.open(OUT.resolve().as_uri())
            opened = True
