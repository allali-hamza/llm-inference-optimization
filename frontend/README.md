# Qwen3 Inference Lab — Streamlit frontend

Client of the `llm-inference-optimization` FastAPI backend. It never loads a model or runs a benchmark.

## Layout

```text
llm-inference-optimization/
├── .streamlit/config.toml      # theme (must sit in the folder you launch from)
├── frontend/
│   ├── app.py  config.py  theme.py  api.py  playground.py  dashboard.py
└── benchmarks/results/{advanced,batching}/*.csv
```

## Run

```bash
uv add "streamlit>=1.40" requests pandas "altair>=5"
uv run uvicorn llm_inference_optimization.api.main:app      # terminal 1
uv run streamlit run frontend/app.py                        # terminal 2
```

Environment variables: `LLM_API_URL` (default `http://127.0.0.1:8000`), `BENCH_RESULTS_DIR` (default `./benchmarks/results`).
