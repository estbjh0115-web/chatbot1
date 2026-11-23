"""Offline, side-effect-free usability helpers for PDF selection, search and evidence audits.

Audit results are conservative heuristics, never factual accuracy scores.
"""
from __future__ import annotations

import re

ANSWER_STYLES = {
    "핵심만": "핵심 답을 2~4문장으로 짧게 정리하라. 필요한 수치와 단위를 유지하라.",
    "자세히": "질문에 답하면서 배경과 조건을 문단과 소제목으로 상세히 설명하라.",
    "표로 정리": "비교 가능한 내용은 Markdown 표로 정리하고, 표 밖에 짧은 결론을 덧붙여라.",
    "발표용": "발표 슬라이드에 쓸 수 있게 핵심 bullet 3~5개와 한 줄 결론으로 정리하라.",
}
NUMBER_PATTERN = re.compile(r"(?<![0-9])\d[\d,]*(?:\.\d+)?(?:\s*[:：]\s*\d+)?\s*%?")


def scope_options(docs: list[dict]) -> dict[str, str]:
    """Map only actual upload hashes to display labels; duplicate filenames are allowed."""
    options = {"all": "전체 PDF"}
    for doc in docs:
        options[doc["id"]] = doc["name"]
    return options


def search_pages(docs: list[dict], keyword: str, selected_doc_id: str = "all", limit: int = 35) -> list[dict]:
    """Local raw-PDF text search, without external AI requests or supplement text."""
    term = keyword.strip().casefold()
    if not term or len(term) > 150:
        return []
    matches = []
    for doc in docs:
        if selected_doc_id != "all" and doc["id"] != selected_doc_id:
            continue
        for num, body in enumerate(doc["pages"], 1):
            pos = body.casefold().find(term)
            if pos < 0:
                continue
            left, right = max(0, pos - 95), min(len(body), pos + len(term) + 130)
            excerpt = body[left:right].replace("\n", " ").strip()
            matches.append({"doc_id": doc["id"], "doc_name": doc["name"],
                            "page": num, "excerpt": ("…" if left else "") + excerpt + ("…" if right < len(body) else "")})
            if len(matches) >= limit:
                return matches
    return matches


def response_claims(answer: str, evidence: list[dict], limit: int = 8) -> list[dict]:
    """Check verifiable numeric tokens against each retrieved evidence block.

    A number appearing in the same evidence block is *not* proof of a claim;
    no semantic entailment, unit consistency, or correctness is certified.
    """
    if not answer or answer.strip().startswith("문서에서 찾지 못했다."):
        return []
    main_text = re.split(r"\n\s*근거 페이지\s*:", answer, maxsplit=1)[0]
    main_text = re.sub(r"\[[^\[\]\n]+? p\.\d+\]", "", main_text)
    statements = [x.strip(" -*#\t") for x in re.split(r"(?<=[.!?。])\s+|\n+", main_text)
                  if x.strip(" -*#\t")]
    results = []
    for sentence in statements[:limit]:
        if len(sentence) < 5 or sentence.startswith("|"):
            continue
        nums = [re.sub(r"\s+", "", x) for x in NUMBER_PATTERN.findall(sentence)]
        if nums:
            found = any(all(n in re.sub(r"\s+", "", chunk.get("text", "")) for n in nums)
                        for chunk in evidence)
            label = "수치 표기 발견 · 의미 검토 필요" if found else "수치 근거 미확인 · 원문 확인"
        else:
            label = "내용 근거 직접 확인 필요"
        results.append({"주장": sentence[:220], "점검 결과": label})
    return results


def answer_markdown(answer: str, evidence: list[dict]) -> bytes:
    """Individual answer export, no API key or PDF raw bytes."""
    sources = list(dict.fromkeys(x.get("source", "") for x in evidence if x.get("source")))
    body = ["# PDF 질의응답 결과", "", answer.strip(), "", "## 검색된 근거 후보"]
    body.extend("- " + s for s in sources) if sources else body.append("- 검색 결과 없음")
    body += ["", "> 주의: 위 페이지는 검색된 근거 후보이며, 답변 내용의 사실 여부를 자동 보증하지 않습니다."]
    return "\n".join(body).encode("utf-8-sig")


def conversation_markdown(messages: list[dict]) -> bytes:
    """Readable full chat export without attached PDF content."""
    lines = ["# EXAONE PDF 대화 기록", "", "문서 원문은 첨부되지 않습니다.", ""]
    for msg in messages:
        lines.extend(["## " + ("질문" if msg.get("role") == "user" else "답변"),
                      "", str(msg.get("content", "")), ""])
    return "\n".join(lines).encode("utf-8-sig")