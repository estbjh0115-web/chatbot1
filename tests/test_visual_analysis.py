"""Offline regression tests for optional local OCR and Gemini visual analysis."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import fitz
from visual_analysis import (
    VisualAnalysisError, VISION_ENDPOINT, page_visual_hints,
    render_page_png, extract_native_tables, extract_page_ocr,
    analyze_page_vision, append_review_draft,
)

ROOT = Path(__file__).resolve().parents[1]


def _pdf():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((48, 48), "2024 TEST CHART", fontsize=15)
    page.draw_rect(fitz.Rect(30, 70, 120, 140))
    data = pdf.tobytes()
    pdf.close()
    return {"name": "demo.pdf", "id": "demo", "bytes": data, "pages": ["2024 TEST CHART"]}


class VisualAnalysisTests(unittest.TestCase):
    def test_pdf_render_and_candidate_hints(self):
        pdf = _pdf()
        self.assertTrue(render_page_png(pdf, 1).startswith(b"\x89PNG"))
        self.assertEqual(page_visual_hints(pdf)[0]["페이지"], 1)
        with self.assertRaises(VisualAnalysisError):
            render_page_png(pdf, 3)

    def test_native_tables_empty_for_no_table(self):
        self.assertEqual(extract_native_tables(_pdf(), 1), "")

    def test_missing_ocr_binary_actionable(self):
        with patch("visual_analysis.ocr_available_languages", return_value=(False, [])):
            with self.assertRaisesRegex(VisualAnalysisError, "tesseract-ocr"):
                extract_page_ocr(_pdf(), 1)

    def test_vision_only_on_explicit_request(self):
        requests = []
        class GoodResponse:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "그래프 증가 · 수치는 판독 불명"}]}}]}
        def fake_post(url, **kwargs):
            requests.append((url, kwargs))
            return GoodResponse()
        with self.assertRaises(VisualAnalysisError):
            analyze_page_vision(b"png", "", "[demo.pdf p.1]", post=fake_post)
        self.assertEqual(requests, [])
        result = analyze_page_vision(b"png", "secret-value", "[demo.pdf p.1]", post=fake_post)
        self.assertIn("판독 불명", result)
        self.assertEqual(requests[0][0], VISION_ENDPOINT)
        self.assertEqual(requests[0][1]["headers"]["x-goog-api-key"], "secret-value")
        self.assertNotIn("secret-value", str(requests[0][1]["json"]))

    def test_remote_auth_error_sanitized(self):
        class Denied:
            status_code = 403
        with self.assertRaises(VisualAnalysisError) as caught:
            analyze_page_vision(b"png", "secret-123", "[demo.pdf p.1]",
                                post=lambda *a, **kw: Denied())
        self.assertNotIn("secret-123", str(caught.exception))

    def test_user_review_stays_separate_until_save(self):
        initial = "기존 기록"
        draft = append_review_draft(initial, "TOP 20킬", "OCR")
        self.assertIn("원문 대조 필요", draft)
        self.assertEqual(append_review_draft(draft, "TOP 20킬", "OCR"), draft)
        self.assertNotEqual(initial, draft)
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(src)
        self.assertIn('save_page_note(note_key, merged)', src)
        self.assertIn('st.session_state["visual_review_approved_page"] = note_key', src)
        self.assertIn('st.session_state.pop(f"visual_review_{note_key}", None)', src)
        self.assertIn('disabled=not (vision_key.strip() and allow_send)', src)
        self.assertIn('if run_ocr or run_table:', src)
        self.assertIn('run_vision_', src)
        self.assertIn('page_visual_hints(doc)', src)


if __name__ == "__main__":
    unittest.main()
