from fastapi import FastAPI
from llm_inference_optimization.api.schema import GenerateRequest, GenerateResponse
from llm_inference_optimization.api.service import generate_response


app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    return generate_response(request=request)