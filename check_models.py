"""
지금 내 API 키로 '실제로 답변이 오는' 모델을 찾는 점검 스크립트
- 후보 모델에 짧은 질문을 한 번씩만 보내고 결과를 표로 보여줍니다. (재시도 없음, 모델당 최대 20초)
- 실행: python check_models.py
"""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

try:
    http = types.HttpOptions(timeout=20_000, retry_options=types.HttpRetryOptions(attempts=1))
except Exception:
    http = types.HttpOptions(timeout=20_000)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options=http)

print(f"{'모델':<26}{'결과':<10}{'시간':>8}   비고")
print("-" * 70)
ok_models = []
for m in CANDIDATES:
    t0 = time.time()
    try:
        r = client.models.generate_content(model=m, contents="1+1은? 숫자만 답해.")
        sec = time.time() - t0
        print(f"{m:<26}{'✅ 성공':<10}{sec:>7.1f}s   답변: {r.text.strip()[:20]}")
        ok_models.append(m)
    except Exception as e:
        sec = time.time() - t0
        code = getattr(e, "code", None) or type(e).__name__
        print(f"{m:<26}{'❌ ' + str(code):<10}{sec:>7.1f}s   {str(e)[:60]}")

print("-" * 70)
if ok_models:
    print(f"👉 .env 의 GEMINI_MODEL 을 '{ok_models[0]}' 로 바꿔서 사용하세요.")
else:
    print("👉 모든 모델이 실패했습니다. 결과 화면을 캡처해서 보내주세요.")
