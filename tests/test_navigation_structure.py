"""Fast static safeguards for the split-page Streamlit user interface.

These tests catch accidental reintroduction of stacked tabs and data-loss
patterns even when the Streamlit UI runtime is unavailable locally.
"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / 'app.py').read_text(encoding='utf-8')
MODULE = ast.parse(SOURCE)
MAIN = next(n for n in MODULE.body if isinstance(n, ast.FunctionDef) and n.name == 'main')


class NavigationTests(unittest.TestCase):
    def test_six_pages_are_distinct(self):
        for label in ('PAGE_CHAT', 'PAGE_AUDIT', 'PAGE_NOTES',
                      'PAGE_BULK', 'PAGE_QUIZ', 'PAGE_SETTINGS'):
            self.assertIn(label, SOURCE)
        self.assertIn('st.radio(', SOURCE)
        self.assertNotIn('st.tabs(', SOURCE)

    def test_chat_has_full_width_until_citation_clicked(self):
        self.assertIn('if focused else (st.container(), None)', SOURCE)
        self.assertIn('if preview_col is not None:', SOURCE)
        self.assertIn('source_page(src, docs)', SOURCE)

    def test_document_stats_are_not_part_of_chat(self):
        self.assertIn('if active_page == PAGE_AUDIT:', SOURCE)
        self.assertIn('col1.metric("PDF 수"', SOURCE)
        self.assertIn('if active_page == PAGE_CHAT:', SOURCE)

    def test_optional_supplement_state_survives_page_navigation(self):
        self.assertIn('supplement_uploads = st.session_state.supplement_file_cache', SOURCE)
        self.assertIn('"supplement_file_cache": []', SOURCE)
        self.assertIn('if active_page == PAGE_NOTES:', SOURCE)
        self.assertIn('st.session_state.supplement_file_cache = incoming', SOURCE)

    def test_manual_edits_archive_conversation_before_switch(self):
        self.assertIn('manual_note_signature = hashlib.sha256(', SOURCE)
        self.assertIn('switch_history(st.session_state, current_version, version_label)', SOURCE)
        # Only explicit user-initiated reset clears an ongoing conversation.
        self.assertEqual(SOURCE.count('st.session_state.messages = []'), 1)

    def test_csv_results_not_reset_on_navigation(self):
        self.assertIn('if csv_file_bytes is not None and csv_upload_signature !=', SOURCE)
        self.assertIn('else st.session_state.csv_upload_signature', SOURCE)

    def test_settings_keep_destructive_action_behind_confirmation(self):
        self.assertIn('if active_page == PAGE_SETTINGS:', SOURCE)
        self.assertIn('disabled=not clear_confirm', SOURCE)
        self.assertIn('if st.button("EXAONE API 연결 테스트"', SOURCE)


if __name__ == '__main__':
    unittest.main()