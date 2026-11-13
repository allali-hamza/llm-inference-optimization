"""Live inference page. All GPU work stays in the FastAPI backend."""

import threading
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from api import APIError, generate
from config import BY_API, FULL, VARIANTS
from theme import page_header

PRESETS = {
    "KV caching": "Explain how KV caching speeds up LLM inference.",
    "Capital": "What is the capital of France?",
    "Code": "Write a Python function that checks whether a string is a palindrome.",
    "Summary": "Summarize the plot of Romeo and Juliet in three sentences.",
}

# (API field, label, unit, higher is better)
METRICS = [
    ("ttft_ms", "First token (TTFT)", "ms", False),
    ("mean_decode_step_latency_ms", "Step latency", "ms", False),
    ("decode_throughput_tokens_s", "Throughput", "tok/s", True),
    ("peak_allocated_vram_mib", "Peak VRAM", "MiB", False),
]


@st.cache_resource
def _gpu_lock():
    """One lock shared by every browser session: the backend has no request queue."""
    return threading.Lock()


def _set_prompt(text):
    st.session_state["prompt"] = text


def _run(api_url, prompt, variant, max_new_tokens, compare):
    lock = _gpu_lock()
    if not lock.acquire(blocking=False):
        st.warning("Another generation is already using the GPU. Wait for it to finish, then retry.")
        return
    try:
        started = time.perf_counter()
        with st.spinner("Loading weights and generating on the backend GPU…"):
            data = generate(api_url, prompt, max_new_tokens, variant, compare)
        wall = time.perf_counter() - started
    except APIError as exc:
        st.error(f"**{exc.title}**" + (f"  \n{exc.detail}" if exc.detail else ""))
        if exc.hint:
            st.caption(exc.hint)
        return
    finally:
        lock.release()

    st.session_state["result"] = {
        "data": data, "wall": wall,
        "request": {"prompt": prompt, "variant": variant, "max_new_tokens": int(max_new_tokens)},
    }
    m, diff = data.get("metrics") or {}, data.get("difference_vs_baseline") or {}
    st.session_state.setdefault("history", []).insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Variant": BY_API[variant].label,
        "New tokens": int(max_new_tokens),
        "Wall (s)": round(wall, 1),
        "TTFT (ms)": m.get("ttft_ms"),
        "Step (ms)": m.get("mean_decode_step_latency_ms"),
        "Tok/s": m.get("decode_throughput_tokens_s"),
        "VRAM (MiB)": m.get("peak_allocated_vram_mib"),
        "Speedup vs BF16 (×)": diff.get("throughput_speedup"),
    })
    del st.session_state["history"][25:]


def _delta(selected, baseline, higher_is_better):
    if not isinstance(selected, (int, float)) or not isinstance(baseline, (int, float)) or not baseline:
        return None, "off"
    pct = (selected - baseline) / baseline * 100
    return f"{pct:+.0f}% vs baseline", ("normal" if higher_is_better else "inverse")


def _race_html(rows):
    top = max(value for _, _, value in rows) or 1
    html = ['<div class="race">']
    for label, color, value in rows:
        width = max(value / top * 100, 2)
        html.append(
            f'<div class="race-row"><div class="race-name">{label}</div>'
            f'<div class="race-track"><div class="race-bar" style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<div class="race-val">{value:,.1f} ms</div></div>'
        )
    html.append("</div>")
    return "".join(html)


def _render_result(result):
    data, request = result["data"], result["request"]
    variant = BY_API.get(data.get("variant")) or BY_API[request["variant"]]
    metrics = data.get("metrics") or {}
    baseline = data.get("baseline_metrics")
    diff = data.get("difference_vs_baseline")

    st.markdown("#### Metrics")
    cols = st.columns(4)
    for col, (key, label, unit, higher) in zip(cols, METRICS):
        value = metrics.get(key)
        shown = f"{value:,.1f} {unit}" if isinstance(value, (int, float)) else "—"
        delta, color = _delta(value, (baseline or {}).get(key), higher)
        col.metric(label, shown, delta=delta, delta_color=color)
    st.caption(
        "Timings cover synchronized model-forward steps only. The mean includes the first step; "
        "tokenization, output assembly and weight loading are excluded. "
        "TTFT is the first forward pass, not end-to-end HTTP latency."
    )

    if baseline and diff:
        speedup = diff.get("decode_latency_speedup") or 0
        verdict = "faster" if speedup >= 1 else "slower"
        st.markdown(f"#### {variant.label} against the baseline")
        st.markdown(f'<div class="race-head">{speedup:.2f}× {verdict}</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="race-note">Mean time per forward step. A shorter bar is faster.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _race_html([
                (BY_API["baseline"].label, BY_API["baseline"].color,
                 baseline.get("mean_decode_step_latency_ms") or 0),
                (variant.label, variant.color, metrics.get("mean_decode_step_latency_ms") or 0),
            ]),
            unsafe_allow_html=True,
        )
        a, b = st.columns(2)
        a.metric("Throughput speedup", f"{diff.get('throughput_speedup', 0):.2f}×")
        vram = diff.get("vram_difference_mib", 0)
        b.metric("VRAM vs baseline", f"{vram:+,.1f} MiB",
                 help="Negative means the selected variant allocated less PyTorch VRAM.")


def render_playground(api_url):
    page_header(
        "Playground",
        "Send one prompt to your Qwen3-1.7B engine and see how each optimization changes speed and memory.",
    )
    st.session_state.setdefault("prompt", PRESETS["KV caching"])

    left, right = st.columns([1, 1.5], gap="large")

    with left:
        st.text_area("Prompt", key="prompt", height=170, placeholder="Ask Qwen3 anything…")
        chips = st.columns(len(PRESETS))
        for chip, (name, text) in zip(chips, PRESETS.items()):
            chip.button(name, key=f"preset_{name}", on_click=_set_prompt, args=(text,), **FULL)

        variant = st.radio(
            "Inference variant",
            [v.api_id for v in VARIANTS],
            index=1,
            format_func=lambda api_id: BY_API[api_id].label,
            captions=[v.blurb for v in VARIANTS],
        )
        max_new_tokens = st.slider("Maximum new tokens", 16, 512, 64, step=16)
        compare = st.toggle(
            "Compare with the BF16 baseline",
            value=variant != "baseline",
            disabled=variant == "baseline",
            help="Runs the baseline right after, so the request takes longer.",
        )
        go = st.button("Generate", type="primary", **FULL)
        st.caption(
            "Greedy decoding, one prompt, no streaming. The backend reloads the weights on every "
            "request, so the wait is longer than the model time shown in the metrics."
        )

    if go:
        prompt = st.session_state["prompt"].strip()
        if not prompt:
            with left:
                st.error("Write a prompt first.")
        else:
            with left:
                _run(api_url, prompt, variant, max_new_tokens, compare)

    result = st.session_state.get("result")

    with right:
        st.markdown("#### Response")
        with st.container(border=True):
            if not result:
                st.markdown("_The answer appears here. Pick a variant, then press **Generate**._")
            else:
                shown = BY_API.get(result["data"].get("variant")) or BY_API[result["request"]["variant"]]
                st.markdown(
                    f'<span class="dot" style="background:{shown.color}"></span><b>{shown.label}</b>',
                    unsafe_allow_html=True,
                )
                st.markdown(result["data"].get("text") or "_The model returned no text._")
        if result:
            st.caption(f"Wall time {result['wall']:.1f} s, including weight loading.")
            st.download_button("Save text", data=result["data"].get("text", ""),
                               file_name="qwen3_response.txt", mime="text/plain", **FULL)

    if result:
        _render_result(result)

    history = st.session_state.get("history")
    if history:
        st.markdown("#### Runs in this session")
        frame = pd.DataFrame(history)
        st.dataframe(frame.round(2), hide_index=True, **FULL)
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("Download CSV", frame.to_csv(index=False), "session_runs.csv", "text/csv", **FULL)
        if c2.button("Clear runs", **FULL):
            st.session_state["history"] = []
            st.session_state.pop("result", None)
            st.rerun()
