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

    def test_search_description_wraps_in_two_lines(self):
        self.assertIn('class="exa-search-intro"', APP)
        self.assertIn('<span>AI API를 호출하지 않습니다.</span>', APP)
        self.assertIn('white-space:normal !important;', CSS)
        self.assertIn('overflow-wrap:anywhere !important;', CSS)
        self.assertIn('.exa-search-intro span', CSS)

    def test_sidebar_scrolls_and_long_text_wraps(self):
        self.assertIn('overflow-y:auto !important;', CSS)
        self.assertIn('max-height:100dvh !important;', CSS)
        self.assertIn('Sidebar fit: keep every menu item readable', CSS)
        self.assertIn('overflow-wrap:anywhere !important;', CSS)
        self.assertIn('text-overflow:clip !important;', CSS)

    def test_page_titles_use_unclipped_dedicated_component(self):
        import re
        self.assertIn('def render_page_title(', APP)
        for title in ('AI 채팅', '문서 내부 검색', '문서 분석', '그림·표 보강',
                      '일괄 질문', '퀴즈', '설정'):
            self.assertIn(f'"{title}")', APP)
        self.assertEqual(APP.count('render_page_title("'), 8)  # seven menus and welcome screen
        self.assertIn('class="exa-page-title"', APP)
        self.assertIn('.exa-page-title-label', CSS)
        self.assertIn('overflow:visible !important;', CSS)
        self.assertIn('min-height:48px;', CSS)
        self.assertIn('line-height:1.5 !important;', CSS)

    def test_welcome_and_page_titles_use_html_unclipped(self):
        self.assertIn('st.html(', APP)
        self.assertIn('class="page-header"', APP)
        self.assertIn('class="exa-page-title"', APP)
        self.assertIn('min-height:98px;', CSS)
        self.assertIn('line-height:1.65 !important;', CSS)
        self.assertIn('max-height:none !important;', CSS)

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
