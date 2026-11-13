"""Shared constants: the five backend variants, their colors, and layout helpers."""

import inspect
from dataclasses import dataclass

import streamlit as st

INK = "#14181F"
INK_SOFT = "#4A5565"
FONT = "Hanken Grotesk, system-ui, sans-serif"


@dataclass(frozen=True)
class Variant:
    api_id: str      # value sent to POST /generate
    csv_id: str      # value found in advanced.csv
    label: str
    blurb: str
    color: str


VARIANTS = [
    Variant("baseline", "baseline_bf16", "Baseline BF16",
            "Recomputes the full sequence at every token. The reference.", "#7A8699"),
    Variant("kv_cache", "kv_cache_bf16", "KV Cache",
            "Reuses past keys and values instead of recomputing them.", "#2F5BFF"),
    Variant("kv_cache_flash_attention", "kv_cache_flash_attention", "KV + FlashAttention",
            "KV Cache with the FlashAttention kernel.", "#0FA3A3"),
    Variant("kv_cache_int8", "kv_cache_int8", "KV + INT8",
            "KV Cache with experimental INT8 linear layers.", "#E59A1F"),
    Variant("kv_cache_int4", "kv_cache_int4", "KV + INT4",
            "KV Cache with experimental INT4 linear layers.", "#C2417C"),
]
BY_API = {v.api_id: v for v in VARIANTS}
BY_CSV = {v.csv_id: v for v in VARIANTS}

# Streamlit renamed use_container_width -> width="stretch"; support both.
_NEW_API = "width" in inspect.signature(st.altair_chart).parameters
FULL = {"width": "stretch"} if _NEW_API else {"use_container_width": True}
