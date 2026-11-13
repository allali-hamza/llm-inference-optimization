from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=64, ge=1)
    variant: Literal[
        "baseline",
        "kv_cache",
        "kv_cache_flash_attention",
        "kv_cache_int8",
        "kv_cache_int4",
    ] = "baseline"
    compare_to_baseline: bool = False


class GenerateResponse(BaseModel):
    variant: str
    text: str
    metrics: dict
    baseline_metrics: dict | None = None
    difference_vs_baseline: dict | None = None
    
