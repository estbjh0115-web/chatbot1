"""Single, cohesive visual style for the EXAONE PDF assistant.

Font loads from the upstream Pretendard CDN; local Korean fallback remains available.
No user PDF or API key is sent to the font provider by this code.
"""

APP_CSS = r"""
<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css");

/* EXAONE design tokens: modern minimal light, Nov 2026 */
:root {
  color-scheme: light !important;
  --exa-bg: #F7F9FC;
  --exa-surface: #FFFFFF;
  --exa-surface-hover: #F2F0FF;
  --exa-text: #1F2937;
  --exa-muted: #586579;
  --exa-primary: #6554D9;
  --exa-primary-hover: #5543C6;
  --exa-primary-light: #F0EDFF;
  --exa-border: #E2E8F0;
  --exa-border-focus: #A79BF1;
  --exa-radius: 12px;
  --exa-font: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif;
}

/* Single light palette: remove prior dark and light CSS collisions. */
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stHeader"] {
  background: var(--exa-bg) !important;
  color: var(--exa-text) !important;
}
html, body, .stApp,
[data-testid="stMain"], [data-testid="stSidebar"],
[data-testid="stMain"] p, [data-testid="stMain"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
button, input, textarea, select {
  font-family: var(--exa-font) !important;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1190px !important;
  padding: 27px 36px 95px !important;
}
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] :is(p,li,strong,em,h1,h2,h3,h4,h5,h6) {
  color: var(--exa-text) !important;
  opacity: 1 !important;
}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stChatMessage"] p {
  font-size: 16px !important;
  line-height: 1.66 !important;
  overflow-wrap: anywhere;
}
[data-testid="stMain"] h1 {font-size:28px !important; font-weight:750 !important; line-height:1.35 !important;}
[data-testid="stMain"] h2 {font-size:22px !important; font-weight:700 !important;}
[data-testid="stMain"] h3 {font-size:20px !important; font-weight:700 !important;}
[data-testid="stMain"] :is(h1,h2,h3,h4,h5,h6) {letter-spacing:-.025em !important;}
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: var(--exa-muted) !important;
  font-size: 13px !important;
  line-height: 1.55 !important;
  opacity: 1 !important;
}
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
  color: var(--exa-text) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
}
[data-testid="stMain"] a, [data-testid="stSidebar"] a {
  color: var(--exa-primary) !important;
}
[data-testid="stMain"] hr, [data-testid="stSidebar"] hr {
  border-color: var(--exa-border) !important;
}

/* Sidebar: navigation is physically before upload and API controls. */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: var(--exa-surface) !important;
  color: var(--exa-text) !important;
}
[data-testid="stSidebar"] {
  width: 310px !important;
  border-right: 1px solid var(--exa-border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] :is(p,strong,h1,h2,h3,h4,h5) {
  color: var(--exa-text) !important;
}
.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  background: #FAFAFF;
  border: 1px solid #EAE6FF;
  border-radius: var(--exa-radius);
  padding: 14px 12px;
  margin: 2px 0 22px;
}
.sidebar-logo .symbol {
  display: grid;
  place-items: center;
  width: 43px;
  height: 43px;
  flex-shrink: 0;
  background: var(--exa-primary);
  color: #FFFFFF !important;
  border-radius: 11px;
  font-size: 21px;
}
.sidebar-logo strong {font-size:16px; font-weight:750; color:var(--exa-text) !important;}
.side-mini {font-size:12px; margin-top:4px; color:var(--exa-muted) !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] {margin-top:3px;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap:5px !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
  display:flex !important; align-items:center !important;
  min-height:46px !important; width:100% !important;
  border-radius:11px !important; box-sizing:border-box !important;
  padding:10px 13px !important; margin:0 !important;
  background:var(--exa-surface) !important;
  border:1px solid transparent !important;
  color:var(--exa-text) !important;
  cursor:pointer !important;
  transition:background .15s ease,border-color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
  background:#F7F6FD !important;
  border-color:var(--exa-border) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background:var(--exa-primary-light) !important;
  border-color:#D5CFFF !important;
  color:#4433A5 !important;
  font-weight:700 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label span {
  font-size:15px !important; font-weight:inherit !important; color:inherit !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:focus-within {
  outline:2px solid var(--exa-primary) !important; outline-offset:2px !important;
}

/* Buttons: 44-46px tap targets, consistent corners and strong contrast. */
[data-testid="stMain"] .stButton button,
[data-testid="stSidebar"] .stButton button,
[data-testid="stMain"] [data-testid="stDownloadButton"] button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button,
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
  min-height:45px !important;
  border-radius:11px !important;
  font-size:14px !important;
  font-weight:650 !important;
  line-height:1.3 !important;
}
[data-testid="stApp"] button[data-testid="stBaseButton-primary"] {
  background:var(--exa-primary) !important;
  border:1px solid var(--exa-primary) !important;
  color:#FFFFFF !important;
}
[data-testid="stApp"] button[data-testid="stBaseButton-primary"]:hover {
  background:var(--exa-primary-hover) !important;
}
[data-testid="stApp"] button[data-testid="stBaseButton-primary"] * {color:#FFFFFF !important;}
[data-testid="stApp"] button[data-testid="stBaseButton-secondary"] {
  background:#FFFFFF !important;
  color:var(--exa-text) !important;
  border:1px solid #CBD5E1 !important;
}
[data-testid="stApp"] button[data-testid="stBaseButton-secondary"]:hover {
  border-color:var(--exa-border-focus) !important;
  background:var(--exa-surface-hover) !important;
}
[data-testid="stApp"] button[data-testid="stBaseButton-secondary"] * {color:var(--exa-text) !important;}
[data-testid="stApp"] button:disabled {opacity:.57 !important; cursor:not-allowed !important;}
[data-testid="stApp"] button:focus-visible,
[data-testid="stApp"] input:focus-visible,
[data-testid="stApp"] textarea:focus-visible {
  outline:2px solid var(--exa-primary) !important;
  outline-offset:2px !important;
}

/* Form fields: readable and aligned across sidebar and main. */
[data-testid="stApp"] [data-testid="stTextArea"] textarea,
[data-testid="stApp"] [data-testid="stTextInput"] input,
[data-testid="stApp"] [data-testid="stNumberInput"] input,
[data-testid="stApp"] [data-testid="stChatInput"] textarea {
  background:#FFFFFF !important;
  color:var(--exa-text) !important;
  border-radius:11px !important;
  font-size:15px !important;
  line-height:1.65 !important;
  caret-color:var(--exa-primary) !important;
}
[data-testid="stApp"] [data-testid="stTextArea"] textarea {
  border:1px solid #CBD5E1 !important;
  padding:12px 14px !important;
}
[data-testid="stApp"] [data-testid="stChatInput"] textarea {font-size:16px !important;}
[data-testid="stApp"] input::placeholder,
[data-testid="stApp"] textarea::placeholder {
  color:#68768A !important; opacity:1 !important;
}
[data-testid="stApp"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stApp"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background:#FFFFFF !important;
  color:var(--exa-text) !important;
  border:1px solid #CBD5E1 !important;
  border-radius:11px !important;
  min-height:45px !important;
}
[data-testid="stApp"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
  color:var(--exa-text) !important;
}
[data-testid="stApp"] [data-testid="stFileUploaderDropzone"] {
  border:1px dashed #BFC9DC !important;
  background:#F8F9FE !important;
  border-radius:var(--exa-radius) !important;
}
[data-testid="stApp"] [data-testid="stFileUploaderDropzone"] * {
  color:var(--exa-text) !important;
}
[data-testid="stApp"] [data-testid="stSlider"] [role="slider"] {
  background:var(--exa-primary) !important;
}
[data-testid="stApp"] [data-testid="stCheckbox"] label {color:var(--exa-text) !important;}

/* Cards, tables, source previews, and chat bubbles */
[data-testid="stMain"] [data-testid="stMetric"],
[data-testid="stMain"] [data-testid="stChatMessage"] {
  background:#FFFFFF !important;
  border:1px solid var(--exa-border) !important;
  border-radius:var(--exa-radius) !important;
  box-shadow:0 3px 11px rgba(31,41,55,.035);
}
[data-testid="stMain"] [data-testid="stMetric"] {padding:18px 20px !important;}
[data-testid="stMain"] [data-testid="stMetricLabel"] *,
[data-testid="stMain"] [data-testid="stMetricDelta"] * {
  color:var(--exa-muted) !important;
}
[data-testid="stMain"] [data-testid="stMetricValue"] * {
  color:var(--exa-text) !important;
  font-variant-numeric:tabular-nums;
}
[data-testid="stMain"] [data-testid="stChatMessage"] {padding:15px 17px !important;}
[data-testid="stMain"] [data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background:#FFFFFF !important;
  border:1px solid var(--exa-border) !important;
  border-radius:var(--exa-radius) !important;
}
[data-testid="stApp"] [data-testid="stExpander"] summary {
  min-height:45px !important;
  color:var(--exa-text) !important;
  font-weight:600 !important;
}
[data-testid="stApp"] [data-testid="stExpander"] summary * {
  color:var(--exa-text) !important;
}
[data-testid="stApp"] [data-testid="stExpander"] details[open] > summary {
  background:#FAFBFE !important;
}
[data-testid="stMain"] [data-testid="stTable"] table,
[data-testid="stMain"] [data-testid="stDataFrame"] {
  background:#FFFFFF !important;
  color:var(--exa-text) !important;
  border-color:var(--exa-border) !important;
  border-radius:var(--exa-radius) !important;
}
[data-testid="stMain"] [data-testid="stTable"] th {
  background:#F0F3FA !important; color:var(--exa-text) !important;
  font-weight:650 !important;
}
[data-testid="stMain"] [data-testid="stTable"] td {
  background:#FFFFFF !important; color:var(--exa-text) !important;
}
[data-testid="stMain"] [data-testid="stCodeBlock"] pre,
[data-testid="stMain"] [data-testid="stCodeBlock"] code {
  font-family:ui-monospace, SFMono-Regular, Consolas, monospace !important;
}
[data-testid="stMain"] [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
  /* Alert backgrounds are controlled by Streamlit's status type. */
  color:var(--exa-text) !important;
}

/* Custom dashboard blocks retained for legacy screens and document overview. */
.page-header {display:flex; align-items:center; gap:14px; margin:6px 0 22px;}
.page-header .page-icon {
  display:grid; place-items:center; flex-shrink:0;
  width:50px; height:50px; border-radius:12px; font-size:26px;
  background:#EDEAFF; color:var(--exa-primary);
}
.page-header .eyebrow {display:none;}
.page-header .heading {font-size:28px; font-weight:750; color:var(--exa-text); line-height:1.3; letter-spacing:-.035em;}
.page-header .subtitle {font-size:13px; color:var(--exa-muted); line-height:1.55; margin-top:4px;}
.welcome-hero {
  display:flex;align-items:center;gap:20px;
  padding:22px 24px;margin:8px 0 22px;
  background:#FFFFFF;border:1px solid var(--exa-border);
  border-radius:var(--exa-radius);
}
.welcome-hero .hero-icon {font-size:34px;padding-right:20px;border-right:1px solid var(--exa-border);}
.welcome-hero .hero-title {font-weight:700;font-size:21px;color:var(--exa-text);line-height:1.4;}
.welcome-hero .hero-desc {font-size:14px;color:var(--exa-muted);line-height:1.6;}
.dashboard-card {
  min-height:164px; padding:20px; margin-bottom:12px;
  background:#FFFFFF; border:1px solid var(--exa-border);
  border-radius:var(--exa-radius);
}
.dashboard-card .card-heading {font-weight:700;font-size:17px;color:var(--exa-text);margin-bottom:14px;}
.dashboard-card .card-empty {
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;
  padding:15px; min-height:100px; border:1px dashed var(--exa-border);
  border-radius:10px;background:#F9FAFE;
}
.dashboard-card .card-empty .big {font-size:24px;}
.dashboard-card .card-empty .empty-line {font-size:13px;color:var(--exa-text);}
.dashboard-card .card-empty .empty-small {font-size:12px;color:var(--exa-muted);}
.dashboard-card .sample {padding:10px 12px;margin:8px 0;background:#F6F5FD;color:var(--exa-text);border-radius:9px;}
.dashboard-footnote {font-size:13px;color:var(--exa-muted);}
@media (max-width:900px) {
  [data-testid="stSidebar"] {width:min(88vw,310px) !important;}
  [data-testid="stMainBlockContainer"] {padding:20px 16px 85px !important;}
  [data-testid="stMain"] h1, .page-header .heading {font-size:24px !important;}
}
@media (prefers-reduced-motion:reduce) {
  [data-testid="stApp"] * {transition:none !important;scroll-behavior:auto !important;}
}
/* Question mark help: works on hover and keyboard focus. */
.exa-help-line {
  display:flex;
  align-items:center;
  gap:9px;
  flex-wrap:wrap;
  margin:16px 0 10px;
  color:#1F2937;
}
.exa-help-title {
  font-size:17px;
  line-height:1.5;
  font-weight:700;
  letter-spacing:-.02em;
}
.exa-help-wrap {
  position:relative;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  cursor:help;
  outline-offset:3px;
  border-radius:50%;
  z-index:10;
}
.exa-help-mark {
  width:20px;
  height:20px;
  border:1px solid #A4ADC0;
  background:#FFFFFF;
  color:#586579;
  font-size:13px;
  line-height:18px;
  font-weight:700;
  border-radius:50%;
  text-align:center;
}
.exa-help-wrap:hover .exa-help-mark,
.exa-help-wrap:focus .exa-help-mark {
  color:#6554D9;
  border-color:#6554D9;
  background:#F0EDFF;
}
.exa-help-popup {
  position:absolute;
  left:0;
  top:calc(100% + 9px);
  width:min(355px, 68vw);
  padding:12px 14px;
  border-radius:11px;
  border:1px solid #D4CCFB;
  background:#FFFFFF;
  color:#1F2937;
  font-family:var(--exa-font);
  font-size:13px;
  font-weight:450;
  line-height:1.6;
  box-shadow:0 9px 22px rgba(31,41,55,.14);
  opacity:0;
  visibility:hidden;
  pointer-events:none;
  transition:opacity .13s ease;
  white-space:normal;
  overflow-wrap:anywhere;
}
.exa-help-wrap:hover .exa-help-popup,
.exa-help-wrap:focus .exa-help-popup,
.exa-help-wrap:focus-within .exa-help-popup {
  opacity:1;
  visibility:visible;
}
@media(max-width:650px) {
  .exa-help-title {font-size:16px;}
  .exa-help-popup {left:auto;right:-20px;width:min(285px,72vw);}
}
</style>
"""