"""
PDF 렌더링 잠금 - pypdfium2 는 여러 요청이 동시에 사용하면 빈 이미지/오류가 날 수 있어
그림 생성은 한 번에 하나씩만 처리하도록 순서를 지킵니다.
"""
import threading

PDF_LOCK = threading.Lock()
