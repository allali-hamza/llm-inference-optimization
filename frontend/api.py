"""HTTP client for the FastAPI inference service (server-to-server, no CORS needed)."""

import requests


class APIError(Exception):
    """Error with a short title, details, and a fix suggestion, shown directly in the UI."""

    def __init__(self, title, detail="", hint=""):
        super().__init__(title)
        self.title, self.detail, self.hint = title, detail, hint


def health_check(base_url):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=2)
        return response.ok and response.json().get("status") == "ok"
    except (requests.RequestException, ValueError):
        return False


def _format_422(detail):
    if isinstance(detail, list):
        parts = []
        for item in detail:
            where = ".".join(str(p) for p in item.get("loc", [])[1:]) or "request"
            parts.append(f"{where}: {item.get('msg', 'invalid value')}")
        return "; ".join(parts)
    return str(detail)


def generate(base_url, prompt, max_new_tokens, variant, compare_to_baseline):
    payload = {
        "prompt": prompt,
        "max_new_tokens": int(max_new_tokens),
        "variant": variant,
        "compare_to_baseline": bool(compare_to_baseline and variant != "baseline"),
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/generate", json=payload,
            timeout=(10, 600),  # weights are reloaded on every request
        )
    except requests.Timeout as exc:
        raise APIError("The request timed out", "No answer after 10 minutes.",
                       "Lower the token limit or turn off the baseline comparison.") from exc
    except requests.RequestException as exc:
        raise APIError("FastAPI is unreachable", str(exc),
                       f"Start the backend and check the address ({base_url}).") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if response.status_code == 422:
            raise APIError("The API rejected the request (422)", _format_422(detail),
                           "Check the prompt and the token limit.")
        raise APIError(
            f"The backend failed (HTTP {response.status_code})", str(detail)[:500],
            "Likely causes: GPU out of memory, CUDA error, or model loading failure. "
            "The uvicorn terminal has the full traceback.",
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise APIError("The API returned invalid JSON") from exc
    if not isinstance(result, dict) or "metrics" not in result or "text" not in result:
        raise APIError("The API response is missing fields", "Expected 'text' and 'metrics'.")
    return result
