"""
Gemini 호출 공통 모듈 - 서버 혼잡(503) 대비 자동 재시도 + 대체 모델 전환

- 503(서버 혼잡), 429(사용량 초과) → 잠시 기다렸다가 재시도
- 같은 모델로 계속 실패 → .env 의 GEMINI_FALLBACK_MODELS 순서대로 다른 모델 시도
- 404(모델 없음) → 바로 다음 모델로 넘어감
"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.6-flash").split(",") if m.strip()]
RETRY_WAITS = [3, 8]       # 재시도 전 대기 시간(초)
TIMEOUT_SEC = 30           # 한 번 요청에 최대 대기 시간(초) - 넘으면 실패로 보고 다음 시도

_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key or "여기에" in key:
            raise SystemExit("❌ .env 파일에 GEMINI_API_KEY 가 없습니다.")
        from google.genai import types
        try:
            # 시간 제한 + 라이브러리 자체 재시도 끄기(재시도는 아래 generate()에서 직접 관리)
            http = types.HttpOptions(timeout=TIMEOUT_SEC * 1000,
                                     retry_options=types.HttpRetryOptions(attempts=1))
        except Exception:
            http = types.HttpOptions(timeout=TIMEOUT_SEC * 1000)
        _client = genai.Client(api_key=key, http_options=http)
    return _client


def _status(e):
    return getattr(e, "code", None) or getattr(e, "status_code", None)


def generate(contents, config=None, verbose=True):
    """generate_content 를 재시도/대체모델과 함께 호출. (응답, 실제 사용한 모델명) 반환"""
    models = [GEMINI_MODEL] + [m for m in FALLBACK_MODELS if m != GEMINI_MODEL]
    last_err = None
    for model in models:
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                if verbose:
                    print(f"  → {model} 호출 중... (최대 {TIMEOUT_SEC}초)")
                res = get_client().models.generate_content(model=model, contents=contents, config=config)
                return res, model
            except Exception as e:
                last_err = e
                code = _status(e)
                if code is None and "timeout" in str(e).lower():
                    code = 504  # 시간 초과도 서버 혼잡과 같이 처리
                if code == 404:
                    if verbose:
                        print(f"  ⚠ {model}: 사용할 수 없는 모델(404) → 다음 모델로")
                    break
                if code in (429, 500, 503, 504) and attempt < len(RETRY_WAITS):
                    wait = RETRY_WAITS[attempt]
                    if verbose:
                        print(f"  ⚠ {model}: 서버 혼잡/제한({code}) → {wait}초 후 재시도 ({attempt + 1}/{len(RETRY_WAITS)})")
                    time.sleep(wait)
                    continue
                if code in (429, 500, 503, 504):
                    if verbose:
                        print(f"  ⚠ {model}: 계속 실패 → 대체 모델로 전환")
                    break
                raise  # 키 오류(400/401/403) 등은 바로 알려줌
    raise RuntimeError(f"모든 모델 호출 실패. 잠시 후 다시 시도하세요. 마지막 오류: {last_err}")