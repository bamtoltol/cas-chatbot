# CAS 기술지원 챗봇 – 백엔드 (과제5)

매뉴얼 PDF를 근거로 Gemini가 기술문의에 답하는 RAG 프로그램입니다.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `step1_test_gemini.py` | Gemini API 연결 테스트 |
| `step2_build_index.py` | 매뉴얼 PDF → 페이지별 텍스트 + 임베딩 인덱스 생성 (처음 1회) |
| `step3_ask.py` | 터미널에서 질문/답변 테스트 |
| `rag.py` | 핵심 로직: 매뉴얼 선택 → 검색 → Gemini 답변 |
| `main.py` | FastAPI 서버 (`POST /chat`) – Lovable 연동용 |
| `data/manual_catalog.csv` | 제품군·지역별 매뉴얼 목록 (매뉴얼 추가 시 한 줄 추가) |
| `manuals/` | 매뉴얼 PDF |

## 실행 순서 (Windows, VS Code 터미널)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        (그 다음 .env 를 열어 API 키 입력)
python step1_test_gemini.py
python step2_build_index.py
python step3_ask.py
uvicorn main:app --reload     (브라우저에서 http://localhost:8000/docs)
```
