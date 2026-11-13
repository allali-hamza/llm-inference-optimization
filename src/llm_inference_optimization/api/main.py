import logging

import torch
from fastapi import FastAPI, HTTPException

from llm_inference_optimization.api.schema import GenerateRequest, GenerateResponse
from llm_inference_optimization.api.service import generate_response


app = FastAPI()
logger = logging.getLogger(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"description": "Invalid generation request."},
        503: {"description": "GPU memory is unavailable."},
        500: {"description": "Inference failed."},
    },
)
def generate(request: GenerateRequest):
    try:
        return generate_response(request=request)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=503,
            detail="GPU out of memory. Reduce max_new_tokens or try a smaller variant.",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=500,
            detail="Generation failed. Check the server logs for details.",
        )
