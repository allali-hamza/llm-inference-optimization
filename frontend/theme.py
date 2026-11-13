"""Global CSS. Colors come from the variant palette in config.py."""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Hanken+Grotesk:wght@400..700&display=swap');

html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Hanken Grotesk', system-ui, sans-serif; }
h1, h2, h3, h4, [data-testid="stMetricValue"], .page-title, .race-head {
    font-family: 'Bricolage Grotesque', 'Hanken Grotesk', system-ui, sans-serif;
    letter-spacing: -.02em; font-weight: 700;
}
.stApp { background: #EEF1F4; color: #14181F; }
[data-testid="stSidebar"] { background: #E3E8EE; border-right: 1px solid #CBD3DD; }
.block-container { max-width: 1320px; padding-top: 2.2rem; padding-bottom: 4rem; }

.page-title { font-size: 2.1rem; margin: 0; line-height: 1.1; }
.page-sub { color: #4A5565; margin: .45rem 0 1.6rem 0; max-width: 62ch; line-height: 1.5; }

[data-testid="stMetric"] {
    background: #fff; border: 1px solid #D5DBE3; border-radius: 8px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: #4A5565; }
[data-testid="stMetricValue"] { font-size: 1.7rem; }

.stButton > button[kind="primary"] {
    background: #14181F; color: #fff; border: 0; border-radius: 8px;
    font-weight: 650; padding: .65rem 1rem;
}
.stButton > button[kind="primary"]:hover { background: #2F5BFF; color: #fff; }
.stButton > button[kind="primary"]:disabled { background: #AEB7C3; color: #fff; }
:focus-visible { outline: 2px solid #2F5BFF !important; outline-offset: 2px; }

.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
.legend-row { margin: 3px 0; font-size: .88rem; }

/* Latency race: the one signature element of the app */
.race-head { font-size: 2.6rem; line-height: 1; margin: 4px 0 2px 0; }
.race-note { color: #4A5565; font-size: .88rem; margin-bottom: 14px; }
.race { display: grid; gap: 10px; margin: 6px 0 4px 0; }
.race-row { display: grid; grid-template-columns: 170px 1fr 96px; align-items: center; gap: 14px; }
.race-name { font-weight: 600; font-size: .92rem; }
.race-track { background: #DCE2E9; border-radius: 4px; height: 24px; }
.race-bar { height: 100%; border-radius: 4px; animation: grow .8s ease-out; }
.race-val { text-align: right; font-weight: 650; font-variant-numeric: tabular-nums; }
@keyframes grow { from { width: 0; } }
@media (prefers-reduced-motion: reduce) { .race-bar { animation: none; } }
@media (max-width: 640px) {
    .race-row { grid-template-columns: 1fr 80px; }
    .race-track { grid-column: 1 / -1; grid-row: 2; }
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(
        f'<h1 class="page-title">{title}</h1><p class="page-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )
