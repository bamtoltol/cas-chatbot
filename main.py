"""
[4단계] FastAPI 서버 - Lovable 화면이 호출할 API
실행: uvicorn main:app --reload
테스트: 브라우저에서 http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from figures import figure_image
from page_images import page_image
from rag import answer

app = FastAPI(title="CAS 기술지원 챗봇 API")

# Lovable(다른 주소의 웹페이지)에서 이 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 시연용. 실제 운영 시에는 Lovable 주소만 허용
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    product: str = "Label Scale"      # Label Scale / Simple Weight Scale
    model: str = "CL5200N"
    country: str = "South Korea"
    firmware: str = "V1.95.1"
    revision: str = "R4043"
    role: str = "User"                # User / Dealer / R&D
    question: str = "용지 위치가 안 맞아요"


@app.get("/")
def health():
    return {"status": "ok", "message": "CAS 기술지원 챗봇 API 동작 중"}


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="문의 내용을 입력해주세요.")
    try:
        result = answer(req.question, product=req.product, model=req.model,
                        country=req.country, firmware=req.firmware,
                        revision=req.revision, role=req.role)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {e}")

    # 근거 페이지마다 이미지 전체 주소 추가 → 화면에서 <img src="..."> 로 바로 표시
    base = str(request.base_url).rstrip("/")
    if request.headers.get("x-forwarded-proto") == "https":   # Railway 등 배포 환경
        base = base.replace("http://", "https://", 1)
    for s in result.get("sources", []):
        s["image_url"] = f"{base}/page-image/{result['manual_id']}/{s['page']}"
    # 답변 속 {{그림:ID}} 자리에 넣을 그림 이미지 주소
    for f in result.get("figures", []):
        f["image_url"] = f"{base}/figure-image/{result['manual_id']}/{f['id']}"
    return result


@app.get("/figure-image/{manual_id}/{fig_id}")
def get_figure_image(manual_id: str, fig_id: str):
    """매뉴얼 속 그림(디스플레이 화면, 사진 등)만 잘라낸 PNG 반환"""
    try:
        return FileResponse(figure_image(manual_id, fig_id), media_type="image/png")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/page-image/{manual_id}/{page}")
def get_page_image(manual_id: str, page: int):
    """매뉴얼 페이지를 PNG 이미지로 반환 (근거 페이지 그림 표시용)"""
    try:
        return FileResponse(page_image(manual_id, page), media_type="image/png")
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
