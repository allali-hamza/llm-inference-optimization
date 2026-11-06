import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from llm_inference_optimization.inference.generation import generate
from llm_inference_optimization.model.layers.decoder import DecoderBlock
from llm_inference_optimization.model.model import CausalLanguageModel, TransformerBackbone
from llm_inference_optimization.model.weights import load_pretrained_weights


def load_qwen_model(checkpoint_dir="checkpoint"):
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())

    with torch.device("meta"):
        decoders = nn.ModuleList([
            DecoderBlock(
                hidden_size=config["hidden_size"],
                h_q=config["num_attention_heads"],
                h_kv=config["num_key_value_heads"],
                teta_base=config["rope_theta"],
                d_h=config["head_dim"],
                Dff=config["intermediate_size"],
            )
            for _ in range(config["num_hidden_layers"])
        ])
        model = CausalLanguageModel(
            TransformerBackbone(
                hidden_size=config["hidden_size"],
                vocab_size=config["vocab_size"],
                decoders=decoders,
            )
        )

    load_pretrained_weights(model, checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)

    return model, tokenizer
