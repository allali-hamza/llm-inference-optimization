from pathlib import Path
from huggingface_hub import snapshot_download

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoint"

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir=CHECKPOINT_DIR,
    allow_patterns=[
        "config.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
    ],
)

print(f"Model downloaded to: {model_path}")