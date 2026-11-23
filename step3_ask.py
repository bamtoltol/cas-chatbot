"""
[3단계] 터미널에서 챗봇에 질문해보기 (과제5 실행 화면 캡처용)
※ 제품군/모델/국가는 실제 서비스에서는 Lovable 화면에서 선택하는 값입니다.
   터미널에는 화면이 없어서 여기서 입력받습니다. (엔터 = 기본값)
실행: python step3_ask.py
"""
import os
import sys

from figures import figure_image
from rag import FIG_MARK, answer


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # Windows 기본 사진 앱으로 열기


print("=" * 60)
print(" CAS 기술지원 챗봇 (터미널 테스트)   종료: q")
print("=" * 60)

product = input("제품군 [1] Label Scale  [2] Simple Weight Scale  (기본 1): ").strip()
product = "Simple Weight Scale" if product == "2" else "Label Scale"
model = input("모델명 (기본 CL5200N): ").strip() or "CL5200N"
country = input("국가 (기본 South Korea): ").strip() or "South Korea"
firmware = input("펌웨어 버전 (기본 V1.95.1): ").strip() or "V1.95.1"

while True:
    q = input("\n❓ 문의 내용: ").strip()
    if q.lower() in ("q", "quit", "exit"):
        break
    if not q:
        continue

    r = answer(q, product=product, model=model, country=country, firmware=firmware)

    # 답변 속 {{그림:ID}} → 그림 파일로 저장하고 위치 표시
    fig_paths = []
    text = r["answer"]
    for fid in FIG_MARK.findall(text):
        if any(f["id"] == fid for f in r["figures"]):
            path = figure_image(r["manual_id"], fid)
            fig_paths.append(path)
            text = text.replace("{{그림:%s}}" % fid, f"   🖼 [그림 {fid}] {path.name}")

    print("\n💬 답변")
    print(text)
    print("\n📄 근거:", ", ".join(
        f"PDF {s['page']}p (매뉴얼 {s['printed_page']}쪽)" for s in r["sources"]) or "없음")
    if r["escalate"]:   # 실제 화면에서는 이때만 '기술문의 접수' 버튼을 보여줌
        print(f"🚩 기술문의 접수 필요: {r['escalate_reason']}")
    print(f"⏱ 응답 시간: {r['elapsed_ms']} ms   🤖 사용 모델: {r.get('model_used', '-')}")

    if fig_paths and input(f"\n   답변 속 그림 {len(fig_paths)}개를 열어볼까요? (y/엔터=건너뛰기): ").strip().lower() == "y":
        for p in fig_paths:
            open_file(p)
