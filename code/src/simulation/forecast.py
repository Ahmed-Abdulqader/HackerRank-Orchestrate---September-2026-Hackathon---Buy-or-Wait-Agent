"""Recurring cash-flow forecasting for the Buy or Wait? simulator.

The dataset mostly contains *historical* events. The ground-truth decisions
assume that established debits (rent, groceries, utilities, subscriptions, ...)
and confirmed salaries continue on their observed cadence into the future, so
`amount_safe_to_pay` reserves those projected outflows too.

This module detects a monthly (or close-to-monthly) recurrence per category
from the settled history before the request date, then projects the median
occurrence forward through the requested horizon.
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

GAP_MIN_DAYS = 5
GAP_MAX_DAYS = 40
HISTORY_WINDOW_DAYS = 400
MIN_OCCURRENCES = 3


def _day(value: str) -> Optional[Any]:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _typical_interval(dates: List[Any]) -> Optional[float]:
    """Returns the median gap between sorted dates if it looks recurring."""
    ordered = sorted(dates)
    gaps = [
        (b - a).days
        for a, b in zip(ordered, ordered[1:])
    ]
    gaps = sorted(gaps)
    if not gaps:
        return None
    med = statistics.median(gaps)
    if not (GAP_MIN_DAYS <= med <= GAP_MAX_DAYS):
        return None
    # Reject irregular cadence: median absolute deviation too large.
    spread = statistics.median([
        abs(g - med) for g in gaps
    ]) if len(gaps) > 1 else 0.0
    if spread > max(3.0, 0.35 * med):
        return None
    return round(med)


def build_recurrence_patterns(
    events: List[Dict[str, Any]], request_date_str: str
) -> List[Dict[str, Any]]:
    """
    Detects recurring per-category patterns from settled history.

    Returns a list of patterns:
        {category, direction, interval_days, amount, last_date}
    where amount is the median of the latest occurring amounts and last_date
    is the most recent settled date for the pattern.
    """
    request_date = _day(request_date_str)
    if request_date is None:
        return []
    window_start = request_date - timedelta(days=HISTORY_WINDOW_DAYS)

    by_key: Dict[Any, List[Dict[str, Any]]] = {}
    for ev in events:
        status = ev.get("status")
        if status != "settled":
            continue
        date = _day(ev.get("settlement_date") or ev.get("event_date"))
        if date is None or not (window_start <= date < request_date):
            continue
        amount = ev.get("amount")
        if amount in (None, ""):
            continue
        key = (ev.get("category"), ev.get("direction"))
        by_key.setdefault(key, []).append({"date": date, "amount": float(amount)})

    patterns = []
    for (category, direction), occ in by_key.items():
        if len(occ) < MIN_OCCURRENCES:
            continue
        dates = sorted(o["date"] for o in occ)
        interval = _typical_interval(dates)
        if interval is None:
            continue
        recent = sorted(occ, key=lambda o: o["date"])[-3:]
        amount = statistics.median(o["amount"] for o in recent)
        patterns.append({
            "category": category,
            "direction": direction,
            "interval_days": interval,
            "amount": round(amount, 2),
            "last_date": dates[-1],
        })

    return patterns


def project_recurring(
    events: List[Dict[str, Any]],
    request_date_str: str,
    end_date: Optional[str] = None,
    spending_changes: Optional[List[Dict[str, Any]]] = None,
    category_include: Optional[set] = None,
    exclude_salary: bool = False,
) -> Dict[Any, float]:
    """
    Projects recurring debits/credits forward into the simulation horizon.

    Existing in-horizon events of the same category+direction within +/-2 days
    of a projected date are considered already present and are not duplicated.

    spending_changes may carry category-level overrides: a reduce_to action on
    a real event of a category also caps that category's forecast amount, and a
    stop action removes the category from the forecast.
    """
    request_date = _day(request_date_str)
    if request_date is None:
        return {}

    horizon = request_date + timedelta(days=90)
    if end_date:
        parsed_end = _day(end_date)
        if parsed_end:
            horizon = max(horizon, parsed_end)

    patterns = build_recurrence_patterns(events, request_date_str)

    # Category-level overrides from spending changes.
    reduce_overrides: Dict[str, float] = {}
    stop_categories: set = set()
    if spending_changes:
        for change in spending_changes:
            category = change.get("_category")
            if change.get("action") == "reduce_to":
                reduce_overrides[category] = float(change.get("new_amount", 0.0) or 0.0)
            elif change.get("action") == "stop":
                stop_categories.add(category)

    # Actual cash-flow dates in the horizon (to avoid duplicating them).
    actual_dates: set = set()
    for ev in events:
        date = _day(ev.get("settlement_date") or ev.get("event_date"))
        if date is not None and request_date <= date <= horizon:
            actual_dates.add(date)

    flows: Dict[Any, float] = {}
    for pat in patterns:
        category = pat["category"]
        direction = pat["direction"]
        if direction != "debit" and not (direction == "credit" and category == "salary"):
            continue
        if exclude_salary and direction == "credit":
            continue
        if category_include is not None and category not in category_include:
            continue
        if stop_categories and category in stop_categories:
            continue
        amount = reduce_overrides.get(category, pat["amount"])

        cursor = pat["last_date"]
        while cursor <= horizon:
            cursor = cursor + timedelta(days=pat["interval_days"])
            if cursor < request_date or cursor > horizon:
                continue
            # Skip if an actual event already covers this occurrence.
            if any(abs((cursor - d).days) <= 2 for d in actual_dates):
                continue
            value = -amount if direction == "debit" else amount
            if cursor >= request_date:
                flows[cursor] = flows.get(cursor, 0.0) + value
    return flows