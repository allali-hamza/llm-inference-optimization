"""Benchmark dashboard backed only by saved CSVs (no GPU calls)."""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from config import BY_CSV, FONT, FULL, INK, INK_SOFT, VARIANTS
from theme import page_header

RESULTS = Path(os.getenv("BENCH_RESULTS_DIR", Path(__file__).resolve().parents[1] / "benchmarks" / "results"))
ADVANCED = RESULTS / "advanced" / "advanced.csv"
BATCHING = RESULTS / "batching" / "batching.csv"

COLOR_SCALE = alt.Scale(domain=[v.label for v in VARIANTS], range=[v.color for v in VARIANTS])
ORDER = [v.label for v in VARIANTS]

METRICS = {
    "Decode throughput": ("throughput_tokens_s", "Tokens / second"),
    "Speedup vs BF16": ("speedup_x", "× baseline throughput"),
    "First forward (TTFT)": ("ttft_ms", "Milliseconds"),
    "Mean step latency": ("mean_token_latency_ms", "Milliseconds / step"),
    "Peak allocated VRAM": ("peak_vram_mib", "MiB"),
}


@st.cache_data(ttl=60, show_spinner=False)
def _read(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _style(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=INK_SOFT, titleColor=INK_SOFT, labelFont=FONT, titleFont=FONT,
                        gridColor="#DDE2E8", domainColor="#B9C1CC", tickColor="#B9C1CC")
        .configure_legend(labelFont=FONT, titleFont=FONT, labelColor=INK)
    )


def _show(chart):
    st.altair_chart(_style(chart), theme=None, **FULL)


def _bar(data, key, title, order):
    base = alt.Chart(data).encode(
        y=alt.Y("variant_label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(f"{key}:Q", title=title, scale=alt.Scale(zero=True)),
    )
    bars = base.mark_bar(size=26, cornerRadiusEnd=3).encode(
        color=alt.Color("variant_label:N", scale=COLOR_SCALE, legend=None),
        tooltip=[alt.Tooltip("variant_label:N", title="Variant"), alt.Tooltip(f"{key}:Q", title=title, format=",.2f")],
    )
    labels = base.mark_text(align="left", dx=6, fontWeight=600, color=INK).encode(
        text=alt.Text(f"{key}:Q", format=",.1f"))
    return (bars + labels).properties(height=max(180, len(data) * 52))


def _tradeoff(data):
    base = alt.Chart(data).encode(
        x=alt.X("peak_vram_mib:Q", title="Peak allocated VRAM (MiB)", scale=alt.Scale(zero=False)),
        y=alt.Y("throughput_tokens_s:Q", title="Tokens / second", scale=alt.Scale(zero=True)),
    )
    points = base.mark_circle(size=230, opacity=.95).encode(
        color=alt.Color("variant_label:N", scale=COLOR_SCALE, legend=None),
        tooltip=[alt.Tooltip("variant_label:N", title="Variant"),
                 alt.Tooltip("throughput_tokens_s:Q", format=",.1f", title="Tok/s"),
                 alt.Tooltip("peak_vram_mib:Q", format=",.0f", title="VRAM (MiB)")],
    )
    names = base.mark_text(dy=-17, fontSize=11, fontWeight=600, color=INK).encode(text="variant_label:N")
    return (points + names).properties(height=max(180, len(data) * 52))


def _grouped(data, key, title, order):
    return (
        alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("variant_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-18)),
            xOffset=alt.XOffset("prompt_label:N"),
            y=alt.Y(f"{key}:Q", title=title, scale=alt.Scale(zero=True)),
            color=alt.Color("prompt_label:N", title="Prompt", scale=alt.Scale(range=["#A9B3C1", INK])),
            tooltip=[alt.Tooltip("variant_label:N", title="Variant"), alt.Tooltip("prompt_label:N", title="Prompt"),
                     alt.Tooltip(f"{key}:Q", title=title, format=",.2f")],
        ).properties(height=260)
    )


def _line(data, key, title):
    return (
        alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=3, color="#2F5BFF").encode(
            x=alt.X("batch_size:O", title="Batch size", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{key}:Q", title=title, scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("batch_size:O", title="Batch size"), alt.Tooltip(f"{key}:Q", title=title, format=",.2f")],
        ).properties(height=230)
    )


def _missing(path):
    st.warning(f"Benchmark file not found: `{path}`")
    st.caption("Set `BENCH_RESULTS_DIR` or place the frontend folder next to `benchmarks/`. No data is invented.")


def _columns_ok(df, required):
    gap = sorted(required - set(df.columns))
    if gap:
        st.error(f"The CSV is missing columns: {gap}")
    return not gap


def _advanced():
    df = _read(str(ADVANCED))
    if df is None:
        return _missing(ADVANCED)
    if not _columns_ok(df, {"variant", "prompt_tokens", "ttft_ms", "mean_token_latency_ms",
                            "throughput_tokens_s", "peak_vram_mib"}):
        return
    df = df.copy()
    df["variant_label"] = df["variant"].map(lambda v: BY_CSV[v].label if v in BY_CSV else v)
    base = df[df["variant"] == "baseline_bf16"].set_index("prompt_tokens")["throughput_tokens_s"]
    df["speedup_x"] = df["throughput_tokens_s"] / df["prompt_tokens"].map(base)
    df["prompt_label"] = df["prompt_tokens"].map(lambda n: f"{int(n)} tokens")

    lengths = sorted(df["prompt_tokens"].dropna().unique().tolist())
    labels = [l for l in ORDER if l in set(df["variant_label"])] + \
             [l for l in df["variant_label"].unique() if l not in ORDER]
    f1, f2 = st.columns([1, 2], gap="large")
    length = f1.radio("Prompt length", lengths, index=len(lengths) - 1, horizontal=True,
                      format_func=lambda n: f"{int(n)} tokens")
    chosen = f2.multiselect("Variants", labels, default=labels)

    pool = df[df["variant_label"].isin(chosen)]
    shown = pool[pool["prompt_tokens"] == length]
    if shown.empty:
        return st.info("Select at least one variant to see results.")

    fastest = shown.loc[shown["throughput_tokens_s"].idxmax()]
    lean = shown.loc[shown["peak_vram_mib"].idxmin()]
    best_speedup = shown["speedup_x"].max()
    k = st.columns(3)
    k[0].metric("Fastest decode", f"{fastest['throughput_tokens_s']:,.1f} tok/s",
                delta=fastest["variant_label"], delta_color="off")
    k[1].metric("Best speedup vs BF16", f"{best_speedup:.2f}×" if pd.notna(best_speedup) else "—",
                delta=None if pd.isna(best_speedup) else shown.loc[shown["speedup_x"].idxmax(), "variant_label"],
                delta_color="off")
    k[2].metric("Lowest peak VRAM", f"{lean['peak_vram_mib']:,.0f} MiB",
                delta=lean["variant_label"], delta_color="off")

    metric = st.radio("Metric", list(METRICS), horizontal=True)
    key, axis = METRICS[metric]
    order = [l for l in labels if l in set(shown["variant_label"])]
    a, b = st.columns([3, 2], gap="large")
    with a:
        st.markdown(f"##### {metric} at {int(length)} tokens")
        _show(_bar(shown, key, axis, order))
    with b:
        st.markdown("##### Speed against memory")
        _show(_tradeoff(shown))
        st.caption("Top-left is best: faster and lighter.")

    if len(lengths) > 1:
        st.markdown(f"##### {metric} by prompt length")
        _show(_grouped(pool, key, axis, order))

    st.markdown("##### Data")
    table = shown[["variant_label", "ttft_ms", "mean_token_latency_ms",
                   "throughput_tokens_s", "peak_vram_mib", "speedup_x"]].copy()
    table.columns = ["Variant", "TTFT (ms)", "Mean step (ms)", "Decode tok/s", "Peak VRAM (MiB)", "Speedup (×)"]
    st.dataframe(table.round(2), hide_index=True, **FULL)
    st.caption("Saved GPU benchmarks, not live API measurements. Throughput = 1000 / mean forward-step latency. "
               "Memory is PyTorch peak allocated VRAM, not total device memory. Quantized variants are experimental "
               "and can be slower despite lower VRAM.")


def _batching():
    df = _read(str(BATCHING))
    if df is None:
        return _missing(BATCHING)
    if not _columns_ok(df, {"batch_size", "ttft_ms", "latency_per_request_ms", "global_throughput_tokens_s",
                            "throughput_per_request_tokens_s", "peak_vram_mib"}):
        return
    df = df.sort_values("batch_size")
    if df.empty:
        return st.info("No batching rows found.")
    first, last = df.iloc[0], df.iloc[-1]
    k = st.columns(3)
    k[0].metric("Aggregate throughput", f"{last['global_throughput_tokens_s']:,.1f} tok/s",
                delta=f"batch size {int(last['batch_size'])}", delta_color="off")
    k[1].metric(f"Gain vs batch size {int(first['batch_size'])}",
                f"{last['global_throughput_tokens_s'] / first['global_throughput_tokens_s']:.2f}×")
    k[2].metric("VRAM at largest batch", f"{last['peak_vram_mib']:,.0f} MiB")
    st.caption("Offline benchmark with KV Cache. The live API accepts a single prompt.")

    charts = [
        ("Aggregate throughput", "global_throughput_tokens_s", "Tokens / second"),
        ("Per-request throughput", "throughput_per_request_tokens_s", "Tokens / second / request"),
        ("First forward (TTFT)", "ttft_ms", "Milliseconds"),
        ("Mean batch step", "latency_per_request_ms", "Milliseconds / batch step"),
        ("Peak allocated VRAM", "peak_vram_mib", "MiB"),
    ]
    cols = st.columns(2, gap="large")
    for i, (title, key, axis) in enumerate(charts):
        with cols[i % 2]:
            st.markdown(f"##### {title}")
            _show(_line(df, key, axis))
    with st.expander("Raw batching data"):
        st.dataframe(df.round(2), hide_index=True, **FULL)
    st.caption("`latency_per_request_ms` in the CSV is the mean time of one decode step for the whole batch.")


def render_dashboard():
    page_header("Benchmarks", "Saved GPU results for every optimization. Nothing here touches the GPU.")
    tab1, tab2 = st.tabs(["Optimizations", "Batching"])
    with tab1:
        _advanced()
    with tab2:
        _batching()
