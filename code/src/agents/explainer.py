import os
import logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ..models.schemas import UsageMetrics
from ..utils.costs import estimate_cost

load_dotenv()
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Pydantic Model for Structured LLM Output
# ──────────────────────────────────────────────
class ExplanationOutput(BaseModel):
    """The LLM only returns the explanation string."""
    decision_explanation: str = Field(
        ...,
        description=(
            "A concise, grounded explanation of the financial decision that uses "
            "concrete figures, dates, and currency amounts."
        ),
    )


# ──────────────────────────────────────────────
# Pydantic AI Agent Initialization
# ──────────────────────────────────────────────
_EXPLAINER_MODEL = "gemini-3.5-flash"
_api_key = os.getenv('GOOGLE_API_KEY')
_explainer_agent = None
if _api_key:
    _explainer_agent = Agent(
        model=GoogleModel(_EXPLAINER_MODEL, provider=GoogleProvider(api_key=_api_key)),
        output_type=ExplanationOutput,
        retries=0,
        system_prompt=(
            "You are a financial communication specialist. Your ONLY job is to write a short, "
            "clear explanation of a financial decision for the user.\n\n"
            "STRICT RULES:\n"
            "1. You will receive pre-computed decision data (status, method, amounts, dates, "
            "payment schedule, spending changes). You must NOT alter, recompute, or contradict "
            "any of these values.\n"
            "2. USE concrete figures: always cite the real currency amounts and dates from the "
            "provided data (e.g. 'Pay IDR 46,018,000 in 3 installments of IDR 15,952,906.67, "
            "starting 8 August 2025'). Conversions to the user's home currency are expected.\n"
            "3. Explain WHY this decision was made, what the user should understand about their "
            "financial position, and any actionable guidance, grounded in the provided numbers.\n"
            "4. If spending changes are required, name the exact changes and the resulting amounts.\n"
            "5. Keep the explanation between 1-3 sentences. Be warm but professional.\n"
            "6. Never hallucinate financial facts not present in the provided data."
        ),
    )
else:
    logger.warning("GOOGLE_API_KEY is not set. Explanations will use the deterministic fallback.")


# ──────────────────────────────────────────────
# Deterministic Fallback Explanation
# ──────────────────────────────────────────────
def _fmt_amount(value: float) -> str:
    """Formats an amount like 25256 -> '25,256' and 25256.5 -> '25,256.5'."""
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return f"{int(rounded):,}"
    return f"{rounded:,.2f}".rstrip("0").rstrip(".")


def _deterministic_explanation(answer_data: Dict[str, Any]) -> str:
    """Grounded, figure-rich fallback so the pipeline never depends on the LLM."""
    method = answer_data.get("method", "unknown")
    amount_safe = answer_data.get("amount_safe_to_pay", 0.0)
    earliest = answer_data.get("earliest_date_for_full_payment")
    schedule = answer_data.get("payment_schedule", [])
    changes = answer_data.get("spending_changes", [])
    currency = answer_data.get("user_home_currency", "USD")
    requested = answer_data.get("requested_amount", 0.0)

    def money(value: float) -> str:
        return f"{_fmt_amount(value)} {currency}"

    if method == "full_payment":
        suffix = ""
        if changes:
            change_text = "; ".join(
                f"stop {c.get('event_id')}"
                if c.get("action") == "stop"
                else f"reduce {c.get('event_id')} to {money(c.get('new_amount', 0))}"
                for c in changes
            )
            suffix = f" Requires {change_text} to stay safe."
        return f"Pay {money(requested)} today." + suffix
    if method == "partial_payment" and len(schedule) >= 2:
        first = schedule[0]
        second = schedule[1]
        return (
            f"Pay {money(first['amount'])} today and the remaining "
            f"{money(second['amount'])} on {second['date']}. "
            "This completes the full request while keeping the minimum balance protected."
        )
    if method == "installments" and schedule:
        count = len(schedule)
        per = money(schedule[0]["amount"])
        start = schedule[0]["date"]
        return (
            f"Use {count} installments of {per}, starting {start}. "
            "This keeps the minimum balance protected."
        )
    if method == "wait":
        fdate = earliest or (schedule[0]["date"] if schedule else "")
        return (
            f"Wait until {fdate}, then pay {money(requested)} in full. "
            "Paying earlier would take the balance below the minimum."
        )
    return (
        f"Do not make this payment by the deadline. Although {money(amount_safe)} is "
        "available today, no safe payment option completes the full request."
    )


# ──────────────────────────────────────────────
# Explanation Function
# ──────────────────────────────────────────────
def generate_explanation(answer_data: Dict[str, Any]) -> Tuple[str, UsageMetrics]:
    """
    Generates ONLY the decision_explanation prose using the LLM, with a
    deterministic, figure-grounded fallback when the API is unavailable.

    The LLM receives all pre-computed decision data and is instructed to use
    concrete amounts and dates while never changing the computed values.

    Args:
        answer_data: dict with keys status, method, amount_safe_to_pay,
            earliest_date_for_full_payment, payment_schedule,
            spending_changes, user_home_currency, requested_amount, request_date.

    Returns:
        Tuple of (explanation_string, UsageMetrics).
    """
    method = answer_data.get("method", "unknown")
    status = answer_data.get("status", "unknown")
    amount_safe = answer_data.get("amount_safe_to_pay", 0.0)
    earliest_date = answer_data.get("earliest_date_for_full_payment")
    schedule = answer_data.get("payment_schedule", [])
    changes = answer_data.get("spending_changes", [])
    currency = answer_data.get("user_home_currency", "USD")
    requested = answer_data.get("requested_amount", 0.0)

    prompt_parts = [
        f"Decision Status: {status}",
        f"Payment Method: {method}",
        f"Requested Amount: {requested:,.2f} {currency}",
        f"Amount Safe to Pay Now: {amount_safe:,.2f} {currency}",
    ]

    if earliest_date:
        prompt_parts.append(f"Earliest Date for Full Payment: {earliest_date}")

    if schedule:
        schedule_desc = "; ".join(
            f"{p.get('date')}: {p.get('amount', 0):,.2f} {currency}" for p in schedule
        )
        prompt_parts.append(f"Payment Schedule: {schedule_desc}")
    else:
        prompt_parts.append("Payment Schedule: none")

    if changes:
        change_desc = "; ".join(
            f"stop {c.get('event_id')}"
            if c.get("action") == "stop"
            else f"reduce {c.get('event_id')} to {c.get('new_amount', 0):,.2f} {currency}"
            for c in changes
        )
        prompt_parts.append(f"Required Spending Changes: {change_desc}")
    else:
        prompt_parts.append("Required Spending Changes: none")

    prompt_parts.append(
        "\nWrite a brief, empathetic explanation for the user using the exact amounts, "
        "dates, and currency from this data. Do not invent new numbers and do not "
        "contradict the decision."
    )

    prompt = "\n".join(prompt_parts)

    try:
        if _explainer_agent is None:
            raise RuntimeError("Explainer agent unavailable (no API key).")
        response = _explainer_agent.run_sync(prompt)
        usage = response.usage
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        metrics = UsageMetrics(
            model_name=_EXPLAINER_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimate_cost(_EXPLAINER_MODEL, input_tokens, output_tokens),
            endpoint="explainer",
        )
        explanation = response.output.decision_explanation.strip()
        logger.info(
            f"Generated explanation ({len(explanation)} chars). "
            f"Tokens: in={input_tokens}, out={output_tokens}"
        )
        return explanation, metrics
    except Exception as exc:
        logger.warning(f"Explainer agent failed ({exc}); using deterministic fallback.")
        return _deterministic_explanation(answer_data), UsageMetrics(
            model_name=_EXPLAINER_MODEL, input_tokens=0, output_tokens=0, cost=0.0,
            endpoint="explainer_fallback",
        )