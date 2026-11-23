# 검증 보고서 (2026-11-23)

대상: `app.py` 1~4단계 통합 + 오류 대처 기능.

## 실측 검증 (현재 실행 환경)

| 항목 | 결과 | 수행 방식 |
|---|---|---|
| Python 문법 컴파일 | 통과 | `python -m py_compile app.py verify_install.py` |
| 기존 1~4단계 회귀 | 통과 | 기존 보강/CSV 관련 17개 테스트 |
| 오류 대처 및 연속 오류 중단 | 통과 | 모의 requests.post 응답으로 401·403·429·5xx/빈 응답/길이 제한/시간초과·SSL 등을 재현 |
| PDF → 페이지별 글자수 → 보강 → 검색 → 답변 → CSV | 통과 | PyMuPDF로 PDF 파일 생성 후 보강과 가짜 EXAONE 응답으로 일괄 질문 2개 처리 |
| 전체 단위/통합 테스트 | **28개 통과** | `python -m unittest discover -s tests -v` |
| PyMuPDF 라이브러리 로딩 | 통과 | 1.26.7 |
| Requests 라이브러리 로딩 | 통과 | 2.32.5 |
| scikit-learn 라이브러리 로딩 | 통과 | 1.8.0 |

## 아직 현장 확인 필요 (미검증)

| 항목 | 이유 | Codespaces 확인 방법 |
|---|---|---|
| Streamlit 실제 서버 구동 | 현재 실행 환경에 미설치, 패키지 서버 DNS 접근 실패 | `python verify_install.py --smoke-server` 후 HTTP 200 확인 |
| GitHub Codespaces 외부 접속 | 해당 사용자의 GitHub 저장소/Codespaces 접근 불가 | Ports → 8501 Open in Browser |
| 실 EXAONE 인증·권한·과금 | 실제 EXAONE API 키 없음 | 앱 사이드바 API 연결 테스트 또는 `python verify_install.py --live` |
| 실 PDF/숨겨진 문제 정답률 | 시험 PDF와 실제 문제 데이터 없음 | 교육 파일과 질문 CSV로 테스트, 원문 근거 대조 |
| GitHub Actions | 사용자의 GitHub 저장소가 아직 연결되지 않음 | 저장소 업로드 후 Actions 탭에서 `Verify PDF chatbot` 녹색 체크 확인 |

**중요:** 모의 API 테스트 성공은 실제 EXAONE 인증 성공을 뜻하지 않습니다. 실 브라우저/네트워크/비밀키 의존성은 사용자 Codespaces에서 직접 확인할 수 있도록 진단 기능을 제공합니다.
