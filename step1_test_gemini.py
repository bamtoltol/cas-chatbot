"""
[1단계] Gemini 연결 테스트
- .env 에 넣은 API 키로 Gemini에 짧은 질문을 보내고 답이 오는지 확인합니다.
- 서버가 붐비면(503) 자동으로 기다렸다가 재시도하고, 안 되면 대체 모델로 시도합니다.
- 실행: python step1_test_gemini.py
"""
from gemini_client import GEMINI_MODEL, FALLBACK_MODELS, generate

print(f"기본 모델: {GEMINI_MODEL} / 대체 모델: {FALLBACK_MODELS}")
print("질문을 보냅니다...")

res, used_model = generate("안녕하세요! 한 문장으로 자기소개 해주세요.")

print(f"✅ Gemini 응답 (사용 모델: {used_model}):")
print(res.text)
