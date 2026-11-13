"""Qwen3 Inference Lab: Streamlit client for the local FastAPI service."""

import os

import streamlit as st

from api import health_check
from config import VARIANTS
from dashboard import render_dashboard
from playground import render_playground
from theme import inject_css

st.set_page_config(page_title="Qwen3 Inference Lab", page_icon="⚡", layout="wide")
inject_css()


@st.cache_data(ttl=5, show_spinner=False)
def _alive(url):
    return health_check(url)


with st.sidebar:
    st.markdown("## Qwen3 Inference Lab")
    st.caption("Qwen3-1.7B, PyTorch, local GPU")
    page = st.radio("Page", ["Playground", "Benchmarks"], label_visibility="collapsed")
    st.divider()
    api_url = st.text_input("FastAPI address", value=os.getenv("LLM_API_URL", "http://127.0.0.1:8000"))
    if _alive(api_url):
        st.success("FastAPI connected")
    else:
        st.warning("FastAPI offline. Benchmarks still work.")
    st.divider()
    st.caption("Variants")
    st.markdown(
        "".join(f'<div class="legend-row"><span class="dot" style="background:{v.color}"></span>{v.label}</div>'
                for v in VARIANTS),
        unsafe_allow_html=True,
    )

if page == "Playground":
    render_playground(api_url)
else:
    render_dashboard()
