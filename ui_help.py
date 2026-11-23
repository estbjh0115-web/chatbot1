"""Accessible hover/focus help for PDF analysis and manual supplement workflows."""
from html import escape


def help_heading(title: str, description: str) -> str:
    """An inline question-mark tooltip. Escaped input avoids HTML injection."""
    safe_title = escape(title, quote=True)
    safe_text = escape(description, quote=True)
    return (
        '<div class="exa-help-line">'
        f'<span class="exa-help-title">{safe_title}</span>'
        f'<span class="exa-help-wrap" tabindex="0" aria-label="도움말: {safe_text}">'
        '<span class="exa-help-mark" aria-hidden="true">?</span>'
        f'<span class="exa-help-popup" role="tooltip">{safe_text}</span>'
        '</span></div>'
    )
