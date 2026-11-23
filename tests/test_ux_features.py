import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ux_features import source_page, contextual_query, conversation_context, version_key, switch_history, export_history

class UXFeatureTests(unittest.TestCase):
    def test_link_only_existing_pdf_and_valid_page(self):
        docs = [{"name": "A.pdf", "id": "1", "pages": ["one", "two"]}]
        self.assertEqual(source_page("[A.pdf p.2]", docs), (docs[0], 2))
        for text in ("[A.pdf p.3]", "[A.pdf p.0]", "[B.pdf p.1]", "[A.pdf p.2] extra"):
            self.assertIsNone(source_page(text, docs))
        self.assertIsNone(source_page("[A.pdf p.1]", docs + [docs[0]]))

    def test_followup_uses_previous_question_for_retrieval(self):
        hist = [{"role": "user", "content": "2024 월즈 결승 상대는?"},
                {"role": "assistant", "content": "BLG입니다."}]
        self.assertIn("2024 월즈", contextual_query("그때 몇 세트까지 갔어?", hist))
        self.assertEqual(contextual_query("2023년 월즈 결승전 우승팀의 이름과 근거를 알려주세요.", hist),
                         "2023년 월즈 결승전 우승팀의 이름과 근거를 알려주세요.")
        self.assertIn("BLG", conversation_context(hist))
        self.assertEqual(conversation_context([]), "")

    def test_chat_switch_preserves_and_restores_version(self):
        state = SimpleNamespace(active_chat_key="", active_chat_label="", chat_archives={}, messages=[], source_focus=None)
        a = version_key("a", "notesA")
        b = version_key("a", "notesB")
        self.assertNotEqual(a, b)
        switch_history(state, a, "A 문서")
        state.messages.append({"role": "user", "content": "질문1"})
        switch_history(state, b, "B 보강버전")
        self.assertEqual(state.messages, [])
        self.assertEqual(state.chat_archives[a]["messages"][0]["content"], "질문1")
        state.messages.append({"role": "user", "content": "질문2"})
        switch_history(state, a, "A 문서")
        self.assertEqual(state.messages[0]["content"], "질문1")
        self.assertEqual(state.chat_archives[b]["messages"][0]["content"], "질문2")
        exported = export_history(state.chat_archives, state.active_chat_key,
                                  state.active_chat_label, state.messages)
        self.assertIn('질문1'.encode(), exported)
        self.assertNotIn(b'EXAONE_API_KEY', exported)

if __name__ == '__main__': unittest.main()

class UXIntegrationTests(unittest.TestCase):
    def test_ask_ai_uses_previous_question_but_valid_pdf_evidence(self):
        import importlib.util
        import types
        from unittest.mock import patch
        if 'streamlit' not in sys.modules:
            sys.modules['streamlit'] = types.ModuleType('streamlit')
        spec = importlib.util.spec_from_file_location('ux_app_integration', Path(__file__).resolve().parents[1] / 'app.py')
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        observed = {}
        def fake_retrieve(question, count):
            observed['retrieval'] = question
            return [{'source':'[guide.pdf p.1]', 'text':'T1과 BLG가 2024 월즈 결승을 치러 T1이 3:2로 승리했습니다.'}]
        def fake_call(key, system, prompt, max_tokens):
            observed['prompt'] = prompt
            return '5세트까지 갔습니다. [guide.pdf p.1]'
        old_state = getattr(app.st, 'session_state', None)
        try:
            app.st.session_state = types.SimpleNamespace(answer_token_limit=4096)
            with patch.object(app, 'retrieve', side_effect=fake_retrieve), patch.object(app, 'call_exaone', side_effect=fake_call):
                answer, evidence = app.ask_ai('fake', '그때 몇 세트까지 갔어?', history=[
                    {'role':'user','content':'2024 월즈 결승 상대는 누구야?'},
                    {'role':'assistant','content':'BLG입니다.'},
                ])
            self.assertIn('2024 월즈', observed['retrieval'])
            self.assertIn('대화 맥락', observed['prompt'])
            self.assertIn('[guide.pdf p.1]', answer)
            self.assertEqual(len(evidence), 1)
        finally:
            if old_state is not None:
                app.st.session_state = old_state