"""Accessible hover/focus help for PDF analysis and manual supplement workflows."""
from html import escape
import re


_CIRCLED_STEPS = "①②③④⑤⑥⑦⑧⑨⑩"


def _help_lines(description: str) -> list[str]:
    """Break circled-number instructions into separate readable lines."""
    normalized = re.sub(r"\s+(?=[②③④⑤⑥⑦⑧⑨⑩])", "\n", description.strip())
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def help_heading(title: str, description: str) -> str:
    """An inline question-mark tooltip. Escape all user-provided HTML."""
    safe_title = escape(title, quote=True)
    safe_text = escape(description, quote=True)
    rendered_lines = []
    for line in _help_lines(description):
        if line[0] in _CIRCLED_STEPS:
            number = escape(line[0])
            body = escape(line[1:].strip(), quote=True)
            rendered_lines.append(
                '<div class="exa-help-step">'
                f'<span class="exa-help-step-number">{number}</span>'
                f'<span class="exa-help-step-text">{body}</span>'
                '</div>'
            )
        else:
            rendered_lines.append(
                f'<div class="exa-help-paragraph">{escape(line, quote=True)}</div>'
            )
    safe_popup = "".join(rendered_lines)
    return (
        '<div class="exa-help-line">'
        f'<span class="exa-help-title">{safe_title}</span>'
        f'<span class="exa-help-wrap" tabindex="0" aria-label="도움말: {safe_text}">'
        '<span class="exa-help-mark" aria-hidden="true">?</span>'
        f'<span class="exa-help-popup" role="tooltip">{safe_popup}</span>'
        '</span></div>'
    )
