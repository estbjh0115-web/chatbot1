"""Codespaces 의존성, 단위 테스트, 서버 구동 및 실 API 선택 검증.

실행 예시:
  python verify_install.py
  python verify_install.py --smoke-server
  EXAONE_API_KEY=... python verify_install.py --live   # 키는 명령줄에 직접 적지 않는 편이 안전함

실제로는 Codespaces Secrets에서 EXAONE_API_KEY를 지정하거나 앱 사이드바에서
'API 연결 테스트' 버튼을 이용해 키 노출 없이 점검하세요.
"""
from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ('streamlit','streamlit'),
    ('PyMuPDF','fitz'),
    ('requests','requests'),
    ('scikit-learn','sklearn'),
]


def check_packages() -> bool:
    ok = True
    print('=== 패키지 설치 점검 ===')
    for dist, module in REQUIRED:
        try:
            value = version(dist)
            import_module(module)
            print(f'[OK] {dist}: {value}')
        except (ImportError, PackageNotFoundError, Exception) as exc:
            # 실 서버의 stacktrace/키/원문을 출력하지 않고 설치 여부만 노출
            print(f'[FAIL] {dist}: 설치 또는 로딩 실패 ({type(exc).__name__})')
            ok = False
    if not ok:
        print('설치 명령: python -m pip install -r requirements.txt')
    return ok


def run_tests() -> bool:
    print('\n=== 1~4단계 / API 오류 단위 테스트 ===',flush=True)
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,check=False)
    return result.returncode == 0


def smoke_server() -> bool:
    """실제 Streamlit localhost 서버가 정상 시작됐는지 HTTP 200으로 검증."""
    import socket
    import contextlib
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    command=[sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),
             '--server.headless=true','--server.address=127.0.0.1',
             f'--server.port={port}','--browser.gatherUsageStats=false']
    print(f'\n=== 로컬 HTTP 상태 검사 (포트 {port}) ===',flush=True)
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, text=True)
    try:
        for attempt in range(36):
            if proc.poll() is not None:
                print('[FAIL] Streamlit 서버가 시작 직후 종료됐습니다.')
                return False
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=2) as response:
                    if response.status==200:
                        print('[OK] Streamlit 서버 HTTP 200 /_stcore/health')
                        print('[INFO] 외부 브라우저 사용 여부는 Codespaces의 PORTS 탭에서 확인해 주세요.')
                        return True
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                pass
            time.sleep(0.5)
        print('[FAIL] Streamlit 서버가 정해진 시간 안에 준비되지 않았습니다.')
        return False
    finally:
        proc.terminate()
        try:proc.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=4)


def live_api_check() -> bool:
    print('\n=== EXAONE 실제 API 연결 확인 ===')
    key=os.environ.get('EXAONE_API_KEY','')
    if not key.strip():
        print('[SKIP] EXAONE_API_KEY 환경변수가 없습니다. Codespaces Secrets로 설정하거나 앱의 API 연결 테스트 버튼을 사용하세요.')
        return False
    try:
        import app
        answer=app.connection_test(key)
        if answer.strip():
            print('[OK] EXAONE 실제 API 응답 수신 (내용 및 API 키는 표시하지 않음)')
            return True
        print('[FAIL] EXAONE 응답이 비어 있습니다.')
        return False
    except Exception as exc:
        # 요청 오류는 안전한 EXAONE 오류만 표시; 임의 예외에는 내부 문자열 출력 금지
        if type(exc).__name__=='ExaoneError':
            print('[FAIL]',str(exc))
        else:
            print('[FAIL] 예기치 못한 API 점검 오류:',type(exc).__name__)
        return False


def main() -> int:
    parser=argparse.ArgumentParser(description='PDF 챗봇 Codespaces 자가 검증')
    parser.add_argument('--smoke-server',action='store_true',help='실 Streamlit HTTP 서버 실행 및 헬스체크')
    parser.add_argument('--live',action='store_true',help='EXAONE_API_KEY 환경변수로 실 API 1회 호출 (사용량 발생)')
    args=parser.parse_args()
    package_ok=check_packages()
    tests_ok=run_tests()
    server_ok=smoke_server() if args.smoke_server and package_ok else None
    api_ok=live_api_check() if args.live and package_ok else None
    print('\n=== 검증 요약 ===')
    print('패키지:', 'PASS' if package_ok else 'FAIL')
    print('단위/통합 테스트:', 'PASS' if tests_ok else 'FAIL')
    print('Streamlit HTTP:', 'NOT RUN' if not args.smoke_server else ('PASS' if server_ok else 'FAIL'))
    print('실 API:', 'NOT RUN' if not args.live else ('PASS' if api_ok else 'FAIL/SKIP'))
    return 0 if package_ok and tests_ok and (not args.smoke_server or server_ok) and (not args.live or api_ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
