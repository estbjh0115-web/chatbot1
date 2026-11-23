"""Regression checks for usability helpers, independent of Streamlit/API."""
import unittest

from supplement_ui import (
    clamp_page, note_page_label, note_draft_text,
    note_has_unsaved_draft, supplement_connection_status,
)


class SupplementUXTests(unittest.TestCase):
    def test_formatted_page_label_does_not_crash(self):
        from supplement_ui import coerce_page_number
        self.assertEqual(coerce_page_number("5페이지 · 보강 없음", 9), 5)
        self.assertEqual(coerce_page_number("5페이지 · 직접 보강 완료", 9), 5)
        self.assertEqual(coerce_page_number("5페이지 · 보강 파일 연결", 9), 5)
        self.assertEqual(coerce_page_number(" 2페이지 ", 9), 2)
        self.assertEqual(coerce_page_number("7", 9), 7)
        self.assertEqual(coerce_page_number(4, 9), 4)
        self.assertEqual(coerce_page_number("999페이지 · 보강 없음", 9), 9)
        self.assertEqual(coerce_page_number("알 수 없는 페이지", 9), 1)
        with self.assertRaises(ValueError):
            coerce_page_number("1페이지", 0)

    def test_previous_next_page_boundaries(self):
        self.assertEqual(clamp_page(1, 7, -1), 1)
        self.assertEqual(clamp_page(7, 7, 1), 7)
        self.assertEqual(clamp_page(3, 7, 1), 4)
        self.assertEqual(clamp_page(3, 7, -1), 2)
        with self.assertRaises(ValueError):
            clamp_page(1, 0, 1)

    def test_page_labels_show_which_content_is_connected(self):
        doc = "pdfhash"
        self.assertIn("보강 없음", note_page_label(1, doc, {}, {}))
        self.assertIn("직접 보강 완료", note_page_label(1, doc, {"pdfhash:1": "수동 내용"}, {}))
        self.assertIn("보강 파일 연결", note_page_label(1, doc, {}, {"pdfhash:1": "파일 내용"}))
        self.assertIn("직접 입력 + 보강 파일", note_page_label(1, doc, {"pdfhash:1": "수동"}, {"pdfhash:1": "파일"}))

    def test_draft_restores_on_page_navigation(self):
        key = "doc:3"
        self.assertEqual(note_draft_text(key, {}, {key: "저장된 값"}), "저장된 값")
        self.assertEqual(note_draft_text(key, {key: "수정 중"}, {key: "저장된 값"}), "수정 중")
        self.assertTrue(note_has_unsaved_draft(key, {key: "수정 중"}, {key: "저장된 값"}))
        self.assertFalse(note_has_unsaved_draft(key, {key: "저장된 값"}, {key: "저장된 값"}))
        self.assertEqual(note_draft_text(key, {key: ""}, {key: "내용"}), "")

    def test_file_connections_and_warnings_are_distinct(self):
        self.assertEqual(supplement_connection_status({}, False), (0, 0, 0))
        report = {"matched_sections": 6, "matched_pages": 5, "warnings": ["없는 파일", "형식 오류"]}
        self.assertEqual(supplement_connection_status(report, True), (6, 5, 2))


if __name__ == "__main__":
    unittest.main()
