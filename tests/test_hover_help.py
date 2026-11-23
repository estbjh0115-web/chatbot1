"""Regression checks for hover help and manual supplement search validation."""
import importlib.util
from pathlib import Path
import types
import unittest

from ui_help import help_heading

ROOT = Path(__file__).resolve().parents[1]


class HoverHelpTests(unittest.TestCase):
    def test_tooltip_escapes_html_and_is_keyboard_focusable(self):
        result = help_heading('페이지 <분석>', 'PDF "보강" & 저장 <script>')
        self.assertIn('aria-label=', result)
        self.assertIn('tabindex="0"', result)
        self.assertIn('role="tooltip"', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertNotIn('<script>', result)

    def test_numbered_help_uses_separate_rows(self):
        description = "① PDF 선택 ② 페이지 이동 ③ 내용 저장"
        result = help_heading("사용법", description)
        self.assertEqual(result.count('class="exa-help-step"'), 3)
        self.assertIn('class="exa-help-step-number">①', result)
        self.assertIn('class="exa-help-step-number">②', result)
        self.assertIn('class="exa-help-step-number">③', result)
        self.assertNotIn("① PDF 선택 ② 페이지 이동", result.split('role="tooltip">')[1])

    def test_page_audit_uses_hover_help(self):
        app = (ROOT / 'app.py').read_text(encoding='utf-8')
        theme = (ROOT / 'design_theme.py').read_text(encoding='utf-8')
        self.assertIn('문서 통계', app)
        self.assertIn('파일별 분석', app)
        self.assertIn('페이지별 글자 수와 원본 미리보기', app)
        self.assertIn('그림·표에 빠진 내용 직접 입력', app)
        self.assertIn('st.markdown(help_heading(', app)
        self.assertIn('.exa-help-wrap:hover .exa-help-popup', theme)


class IndexedManualNoteTests(unittest.TestCase):
    def test_long_file_note_does_not_hide_manual_annotation(self):
        import sys
        if 'streamlit' not in sys.modules:
            sys.modules['streamlit'] = types.ModuleType('streamlit')
        spec = importlib.util.spec_from_file_location('app_note_test', ROOT / 'app.py')
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Full-length uploaded note used to make prefix-only verification fragile.
        doc = {'id': 'doc-1', 'name': 'LoL.pdf', 'pages': ['원본: T1 기록']}
        note_key = 'doc-1:1'
        note_text = '월즈 결승 점수: T1 3대2 BLG'
        current = getattr(app.st, 'session_state', None)
        state = types.SimpleNamespace(
            docs=[doc],
            visual_notes={},
            file_notes={note_key: '다른 보강 파일 정보 ' * 170},
            note_drafts={note_key: note_text},
            note_save_status=None,
            chunks=[], vectorizer=None, matrix=None,
            csv_results=[1], csv_errors=[1], csv_finished=True,
            bulk_results=[1], quiz=[1],
        )
        app.st.session_state = state
        try:
            result = app.save_page_note(note_key, note_text)
            self.assertTrue(result['indexed'])
            self.assertEqual(result['chars'], len(note_text))
            self.assertEqual(state.visual_notes[note_key], note_text)
            self.assertFalse(note_key in state.note_drafts)
            manual = [c for c in state.chunks if c.get('note_kind') == 'manual']
            uploaded = [c for c in state.chunks if c.get('note_kind') == 'file']
            self.assertTrue(manual)
            self.assertTrue(uploaded)
            self.assertEqual(manual[0]['source'], '[LoL.pdf p.1]')
            self.assertEqual(manual[0]['note_key'], note_key)
            self.assertTrue(state.vectorizer is not None)

            # Delete should rebuild the index without a stale manual reference.
            cleared = app.save_page_note(note_key, '   ')
            self.assertEqual(cleared['chars'], 0)
            self.assertFalse([c for c in state.chunks if c.get('note_kind') == 'manual'])
            self.assertTrue([c for c in state.chunks if c.get('note_kind') == 'file'])
        finally:
            if current is not None:
                app.st.session_state = current


if __name__ == '__main__':
    unittest.main()
