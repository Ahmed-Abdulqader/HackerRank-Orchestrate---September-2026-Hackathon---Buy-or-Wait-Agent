import logging
from typing import Dict, Any, List
from datetime import datetime

from ..models.schemas import Answer, Payment, SpendingChange

logger = logging.getLogger(__name__)


def _format_amount(value: float) -> str:
    """Renders amounts like 25256 -> '25256' and 620.4 -> '620.40'."""
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}"


def serialize_payments(payments: List[Payment]) -> str:
    """Serializes payments to a pipe-delimited 'YYYY-MM-DD:amount' string, or 'none'."""
    if not payments:
        return "none"
    parts = []
    for payment in sorted(payments, key=lambda p: p.date):
        parts.append(f"{payment.date}:{_format_amount(payment.amount)}")
    return "|".join(parts)


def serialize_changes(changes: List[SpendingChange]) -> str:
    """Serializes spending changes to 'stop:event_id' / 'reduce_to:event_id:amount', or 'none'."""
    if not changes:
        return "none"
    parts = []
    for change in changes:
        if change.action == "stop":
            parts.append(f"stop:{change.event_id}")
        elif change.action == "reduce_to":
            amount = _format_amount(change.new_amount if change.new_amount is not None else 0.0)
            parts.append(f"reduce_to:{change.event_id}:{amount}")
    return "|".join(parts)


def validate_and_format_answer(answer: Answer) -> Dict[str, Any]:
    """
    Validates the final Answer object and serializes it into the exact flat
    dictionary required for output.csv.

    - Floats are rounded to 2 decimal places.
    - Dates are strictly YYYY-MM-DD.
    - payment_plan and spending_changes are pipe-delimited strings (never JSON).
    - No negative amounts or invalid states.
    """
    if answer.amount_safe_to_pay < 0:
        logger.warning("amount_safe_to_pay was negative. Clamping to 0.0.")
        answer.amount_safe_to_pay = 0.0

    if answer.earliest_date_for_full_payment:
        try:
            dt = datetime.strptime(answer.earliest_date_for_full_payment, "%Y-%m-%d")
            answer.earliest_date_for_full_payment = dt.strftime("%Y-%m-%d")
        except ValueError:
            logger.error(
                f"Invalid date format for earliest_date_for_full_payment: "
                f"{answer.earliest_date_for_full_payment}"
            )
            answer.earliest_date_for_full_payment = None

    return {
        "amount_safe_to_pay": _format_amount(answer.amount_safe_to_pay),
        "affordability_status": answer.affordability_status,
        "recommended_payment_method": answer.recommended_payment_method,
        "payment_plan": serialize_payments(answer.payment_schedule),
        "earliest_date_for_full_payment": answer.earliest_date_for_full_payment or "",
        "spending_changes_needed": serialize_changes(answer.spending_changes),
        "decision_explanation": answer.decision_explanation,
    }