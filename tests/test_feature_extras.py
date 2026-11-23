import unittest
from unittest.mock import patch

from feature_extras import (ANSWER_STYLES, answer_markdown, conversation_markdown,
                            response_claims, scope_options, search_pages)


class FeatureExtraTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {"id": "A", "name": "T1.pdf", "pages": ["2024 T1 결승 상대는 BLG이다. 점수는 3:2다.", "라인 운영은 맵을 본다."]},
            {"id": "B", "name": "LCK.pdf", "pages": ["2023 시즌 LCK 개막", "2024 페이커 인터뷰"]},
        ]

    def test_document_scope_and_no_cross_document_hits(self):
        self.assertEqual(scope_options(self.docs)["A"], "T1.pdf")
        self.assertEqual(len(search_pages(self.docs, "2024")), 2)
        hits = search_pages(self.docs, "2024", selected_doc_id="A")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["page"], 1)

    def test_search_casefold_limits_and_empty(self):
        self.assertEqual(search_pages(self.docs, "blg")[0]["doc_id"], "A")
        self.assertEqual(search_pages(self.docs, " "), [])
        self.assertEqual(len(search_pages(self.docs, "2024", limit=1)), 1)

    def test_audit_numeric_match_not_accuracy_claim(self):
        result = response_claims("T1은 2024년에 BLG에 3:2로 승리했다.",
                                 [{"source": "[T1.pdf p.1]", "text": self.docs[0]["pages"][0]}])
        self.assertIn("수치 표기 발견", result[0]["점검 결과"])
        self.assertIn("의미 검토", result[0]["점검 결과"])
        result = response_claims("T1은 2025년에 BLG에 3:2로 승리했다.",
                                 [{"source": "[T1.pdf p.1]", "text": self.docs[0]["pages"][0]}])
        self.assertIn("미확인", result[0]["점검 결과"])
        self.assertEqual(response_claims("문서에서 찾지 못했다.", []), [])

    def test_export_and_styles(self):
        self.assertEqual(len(ANSWER_STYLES), 4)
        result = answer_markdown("T1 3:2", [{"source": "[T1.pdf p.1]", "text": "T1"}]).decode('utf-8-sig')
        self.assertIn("[T1.pdf p.1]", result)
        self.assertIn("자동 보증하지", result)
        result2 = conversation_markdown([{"role": "user", "content": "질문"},
                                         {"role": "assistant", "content": "답"}]).decode('utf-8-sig')
        self.assertIn("## 질문", result2)
        self.assertIn("## 답변", result2)


if __name__ == "__main__":
    unittest.main()

class ScopedRetrievalIntegrationTests(unittest.TestCase):
    def test_scoped_retrieval_excludes_other_pdf_and_scope_is_used_by_ai(self):
        import importlib.util
        import sys
        import types
        from pathlib import Path
        from unittest.mock import patch
        if 'streamlit' not in sys.modules:
            sys.modules['streamlit'] = types.ModuleType('streamlit')
        spec = importlib.util.spec_from_file_location('app_scoped_test', Path(__file__).resolve().parents[1] / 'app.py')
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        documents = [
            {'id': 'A', 'name': 'A.pdf', 'pages': ['페이커 2024 BLG 결승전 우승']},
            {'id': 'B', 'name': 'B.pdf', 'pages': ['페이커 2024 T1의 참가 정보']},
        ]
        old = getattr(app.st, 'session_state', None)
        app.st.session_state = types.SimpleNamespace(docs=documents, visual_notes={}, file_notes={},
                                                      chunks=[], vectorizer=None, matrix=None,
                                                      answer_token_limit=4096)
        try:
            app.set_index()
            all_hits = app.retrieve('페이커 2024', count=10)
            self.assertEqual(len(all_hits), 2)
            specific_hits = app.retrieve('페이커 2024', count=10, allowed_doc_id='A')
            self.assertEqual(len(specific_hits), 1)
            self.assertEqual(specific_hits[0]['doc_id'], 'A')
            self.assertEqual(app.retrieve('페이커 2024', allowed_doc_id='missing'), [])
            with patch.object(app, 'call_exaone', return_value='BLG와 결승전 [A.pdf p.1]') as fake_api:
                answer, evidence = app.ask_ai('dummy', '페이커 2024',
                                              allowed_doc_id='A', answer_style='표로 정리')
                self.assertIn('Markdown 표', fake_api.call_args.args[2])
                self.assertEqual({e['doc_id'] for e in evidence}, {'A'})
                self.assertIn('[A.pdf p.1]', answer)
        finally:
            if old is not None:
                app.st.session_state = old