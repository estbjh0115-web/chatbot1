"""Pure, testable helpers for the PDF page annotation interface."""
from __future__ import annotations


def clamp_page(current: int, total: int, step: int) -> int:
    """Return a 1-based page number inside the selected PDF."""
    if total < 1:
        raise ValueError("PDF에 페이지가 없습니다.")
    return max(1, min(total, int(current) + int(step)))



def coerce_page_number(value: object, total: int, default: int = 1) -> int:
    """Interpret a Streamlit selectbox value as a safe 1-based page number.

    Streamlit widget state may be a page integer or a formatted label such as
    '5페이지 · 보강 없음'. This function accepts both without losing the page.
    """
    import re

    if total < 1:
        raise ValueError("PDF에 페이지가 없습니다.")
    if isinstance(value, bool):
        number = default
    elif isinstance(value, int):
        number = value
    elif isinstance(value, str):
        match = re.match(r"^\s*(\d+)\s*(?:페이지(?:\s*·.*)?|$)", value)
        number = int(match.group(1)) if match else default
    else:
        number = default
    return max(1, min(total, int(number)))


def note_page_label(page: int, doc_id: str, manual: dict, attached: dict) -> str:
    """Compact dropdown status; manual text and uploaded attachment are separate."""
    key = f"{doc_id}:{page}"
    has_manual = bool(str(manual.get(key, "")).strip())
    has_attached = bool(str(attached.get(key, "")).strip())
    if has_manual and has_attached:
        label = "직접 입력 + 보강 파일"
    elif has_manual:
        label = "직접 보강 완료"
    elif has_attached:
        label = "보강 파일 연결"
    else:
        label = "보강 없음"
    return f"{page}페이지 · {label}"


def note_draft_text(note_key: str, drafts: dict, saved: dict) -> str:
    """A draft, even an intentionally cleared empty draft, overrides saved text."""
    return drafts[note_key] if note_key in drafts else saved.get(note_key, "")


def note_has_unsaved_draft(note_key: str, drafts: dict, saved: dict) -> bool:
    """Only changed drafts are considered unsaved."""
    return note_key in drafts and drafts[note_key] != saved.get(note_key, "")


def supplement_connection_status(report: dict, has_upload: bool) -> tuple[int, int, int]:
    """Return (connected sections, connected pages, advisory count)."""
    if not has_upload:
        return (0, 0, 0)
    return (
        int(report.get("matched_sections", 0)),
        int(report.get("matched_pages", 0)),
        len(report.get("warnings", [])),
    )
