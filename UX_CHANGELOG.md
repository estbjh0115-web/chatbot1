# EXAONE PDF 챗봇 - 사용성 개선 (2026-11-23)

## 이번 개선

- 기본 화면을 AI 채팅으로 고정하고 왼쪽 메뉴 6개로 기능별 분리
- PDF 추가 + API 키 입력만 처음부터 노출, 고급 설정/테스트/초기화는 설정 화면으로 이동
- PDF 수, 페이지 수, 글자 수와 페이지별 진단은 문서 분석 메뉴에서만 표시
- 그림/표 보강 TXT/MD 업로드와 수동 보강을 하나의 화면에 통합
- PDF 보강 파일 이름·내용을 메뉴 이동 후 세션 캐시에 유지
- CSV 질문 업로드·실행·결과 다운로드를 일괄 질문 페이지에서 처리
- 답변을 읽을 때 PDF 출처를 클릭해야만 페이지 미리보기 영역 열림
- 이전 대화 및 후속 질문 맥락 유지, 수동 보강 시에도 버전별 이전 대화 보관
- 결과 초기화에는 확인 체크박스 요구
- 기존 밝은 테마 유지

## 확인한 테스트

- 기존 단위/통합 테스트 32개 통과
- 신규 화면 구조 점검 7개 통과
- Python 문법 검사 통과
- EXAONE 실제 키/API 호출 및 Codespaces 브라우저 시각 검증은 별도 필요

## 실행 명령

```bash
cd /workspaces/chatbot1
# 실행 중이라면 Ctrl+C

git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

*8501 포트가 이미 사용 중이라면 기존 Streamlit 서버를 종료하거나 8502 포트를 이용하세요.*

## 한계와 안내

대화 내역 및 업로드한 보강 파일은 Streamlit 세션 메모리에 남습니다. 브라우저 세션이 완전히 종료되면 사라질 수 있으므로 대화 내역 JSON 파일을 다운로드하세요. API 키는 GitHub 코드에 저장하지 마세요. 사내 승인 없이 대외 API로 자료를 전송하지 마세요.