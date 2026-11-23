"""Pure helpers for context-aware retrieval, source navigation, and chat snapshots.

This file has no Streamlit dependency so it can be regression-tested offline.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SOURCE_RE = re.compile(r"^\[(?P<name>.+) p\.(?P<page>[1-9][0-9]*)\]$")
FOLLOWUP_RE = re.compile(
    r"그때|그럼|그러면|그것|그거|그 경기|그 선수|그 사람|그 팀|거기|이어서|앞서|방금|"
    r"그 중|그중|그 이유|왜|몇 세트|몇 경기|몇 명|몇 번|그 당시|그 결과|더 자세히"
)


def source_page(source: str, docs: list[dict]) -> tuple[dict, int] | None:
    """Accept only an actual uploaded document and an in-range PDF page."""
    match = SOURCE_RE.fullmatch(source.strip())
    if not match:
        return None
    name, page = match.group("name"), int(match.group("page"))
    matches = [doc for doc in docs if doc["name"] == name and 1 <= page <= len(doc["pages"])]
    return (matches[0], page) if len(matches) == 1 else None


def _recent_user_messages(history: list[dict], limit: int = 2) -> list[str]:
    questions = [str(msg.get("content", "")).strip() for msg in history
                 if msg.get("role") == "user" and str(msg.get("content", "")).strip()]
    return questions[-limit:]


def contextual_query(question: str, history: list[dict] | None = None) -> str:
    """Include recent questions for short or referential follow-ups only."""
    question = question.strip()
    previous = _recent_user_messages(history or [])
    if not previous or not FOLLOWUP_RE.search(question):
        return question
    return "\n".join(["이전 질문: " + x[:280] for x in previous] + ["현재 질문: " + question])


def conversation_context(history: list[dict] | None = None, limit: int = 6) -> str:
    """Provide short recent context to disambiguate follow-ups; not an answer source."""
    if not history:
        return ""
    lines = []
    for message in history[-limit:]:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = str(message.get("content", "")).strip()
        if not content or content.startswith("요청 실패:"):
            continue
        lines.append(("사용자" if role == "user" else "이전 답변") + ": " + content[:450])
    return "\n".join(lines)[-2200:]


def version_key(pdf_signature: str, supplement_signature: str) -> str:
    """Stable per-document+supplement version, without exposing raw file hashes in UI."""
    value = pdf_signature + "\x1f" + supplement_signature
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:18]


def switch_history(state: Any, new_key: str, new_label: str) -> None:
    """Archive prior session messages and restore messages when revisiting a version.

    A PDF/supplement change does not destroy the previous Q&A. State is in-memory
    only; it is intentionally not written to a shared server disk.
    """
    old_key = getattr(state, "active_chat_key", "")
    if old_key == new_key:
        return
    archives = getattr(state, "chat_archives", None)
    if archives is None:
        archives = {}
        state.chat_archives = archives
    if old_key:
        archives[old_key] = {
            "label": getattr(state, "active_chat_label", "이전 문서"),
            "messages": [dict(m) for m in getattr(state, "messages", [])],
        }
    previous = archives.get(new_key)
    state.messages = [dict(m) for m in previous["messages"]] if previous else []
    state.active_chat_key = new_key
    state.active_chat_label = new_label
    state.source_focus = None


def export_history(archives: dict, active_key: str, active_label: str,
                   active_messages: list[dict]) -> bytes:
    """Portable, user-initiated JSON export; never include an API key or PDF bytes."""
    data = {}
    for key, item in archives.items():
        if isinstance(item, dict) and isinstance(item.get("messages"), list):
            data[key] = {"label": str(item.get("label", "이전 문서")),
                         "messages": item["messages"]}
    if active_key:
        data[active_key] = {"label": active_label, "messages": active_messages}
    return json.dumps({"format": "exaone-chat-history-v1", "conversations": data},
                      ensure_ascii=False, indent=2).encode("utf-8-sig")