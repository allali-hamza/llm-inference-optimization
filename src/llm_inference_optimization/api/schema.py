from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens : int = 64
    variant: str = "baseline"
    compare_to_baseline: bool = False

class GenerateResponse(BaseModel):
    variant: str
    text: str
    metrics: dict
    baseline_metrics: dict | None = None
    difference_vs_baseline: dict | None = None
    
