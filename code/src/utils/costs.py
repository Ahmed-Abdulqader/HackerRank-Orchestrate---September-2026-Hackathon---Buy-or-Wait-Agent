import os

MODEL_PRICING = {
    "z-ai/glm-4.5": {
        "input_per_million": 0.60,
        "output_per_million": 2.20,
    },
    "z-ai/glm-4.5v": {
        "input_per_million": 0.60,
        "output_per_million": 1.80,
    },
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def price_for_model(model_name: str) -> dict:
    if model_name in MODEL_PRICING:
        pricing = MODEL_PRICING[model_name]
        return {
            "input": _env_float(f"PRICE_IN_{model_name.replace('/', '_')}", pricing["input_per_million"]),
            "output": _env_float(f"PRICE_OUT_{model_name.replace('/', '_')}", pricing["output_per_million"]),
        }
    return {"input": _env_float("PRICE_IN_DEFAULT", 0.60), "output": _env_float("PRICE_OUT_DEFAULT", 2.20)}


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimates API cost in USD for a single model call using per-million-token pricing."""
    if model_name == "none" or model_name.startswith("rapidocr"):
        return 0.0
    pricing = price_for_model(model_name)
    return round(
        (input_tokens or 0) / 1_000_000.0 * pricing["input"]
        + (output_tokens or 0) / 1_000_000.0 * pricing["output"],
        6,
    )