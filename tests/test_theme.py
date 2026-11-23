"""Ensure the one-source Pretendard theme keeps the requested design tokens."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
CSS = (ROOT / "design_theme.py").read_text(encoding="utf-8")
CONFIG = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")


class ThemeTests(unittest.TestCase):
    def test_theme_is_single_source_and_python_valid(self):
        ast.parse(APP)
        ast.parse(CSS)
        self.assertIn("from design_theme import APP_CSS", APP)
        self.assertEqual(APP.count("st.markdown(APP_CSS, unsafe_allow_html=True)"), 1)
        self.assertNotIn("UI_CSS =", APP)
        self.assertNotIn("POLISHED_CSS =", APP)

    def test_requested_palette_and_fallbacks(self):
        for token in ("#F7F9FC", "#1F2937", "#6554D9", "#FFFFFF"):
            self.assertIn(token, CSS)
            self.assertIn(token, CONFIG)
        self.assertIn("Pretendard Variable", CSS)
        self.assertIn("Noto Sans KR", CSS)
        self.assertIn("Malgun Gothic", CSS)
        self.assertIn('base = "light"', CONFIG)

    def test_buttons_and_sidebar_navigation(self):
        self.assertIn("min-height:45px", CSS)
        self.assertIn("border-radius:11px", CSS)
        self.assertIn("label:has(input:checked)", CSS)
        self.assertIn("focus-visible", CSS)
        self.assertIn("max-width:900px", CSS)
        sidebar = APP[APP.index("with st.sidebar:"):]
        self.assertLess(sidebar.index("active_page = st.radio("),
                        sidebar.index("uploaded = st.file_uploader("))

    def test_no_functionality_removed(self):
        for keyword in (
            "if active_page == PAGE_NOTES:",
            "if active_page == PAGE_QUIZ:",
            "if active_page == PAGE_BULK:",
            "if active_page == PAGE_CHAT:",
            "save_page_note(",
            "switch_note_page(",
        ):
            self.assertIn(keyword, APP)


if __name__ == "__main__":
    unittest.main()
