"""
Main Orchestrator CLI for "Buy or Wait?" AI Financial Decision Agent.

Reads dataset/ (read-only), rebuilds a local SQLite context in build/,
processes every request in requests.csv, and writes schema-compliant
predictions to output.csv in the project root.

Output columns (exact order):
    request_id, amount_safe_to_pay, affordability_status,
    recommended_payment_method, payment_plan,
    earliest_date_for_full_payment, spending_changes_needed,
    decision_explanation
"""

import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from src.database.build_context import AgentContextDB, ContextNotFoundError
from src.simulation.currency import CurrencyConverter
from src.simulation.simulator import CashFlowSimulator
from src.simulation.ranker import PlanRanker
from src.agents.extractor import extract_facts_from_context
from src.agents.explainer import generate_explanation
from src.agents.image_handler import process_document
from src.utils.validator import validate_and_format_answer
from src.models.schemas import Answer, Payment, SpendingChange

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BUILD_DIR = PROJECT_ROOT / "build"
DB_PATH = BUILD_DIR / "agent_context.db"
OUTPUT_CSV = PROJECT_ROOT / "output.csv"

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
BUILD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(BUILD_DIR / "agent_execution.log"), mode="w"),
    ],
)
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]

_AMOUNT_RE = re.compile(
    r"([A-Za-z0-9#@.:&/\\\s-]{0,36})([\d][\d,]*\.?\d*)"
)

_MONEY_LABEL_SPECIFIC = re.compile(
    r"(balance|due|payable|outstanding|received|paid|refund)", re.IGNORECASE
)
_MONEY_LABEL_GENERAL = re.compile(
    r"(total|amount|bill|charge|fee|rent|sum|price|cost|invoice)", re.IGNORECASE
)
_PHONE_LABEL = re.compile(
    r"(phon|tel|contact|mobile|mob|[.;:]\s*no|reg\.?|uid|patient|admission|address)", re.IGNORECASE
)

_OCR_TEXT_CACHE: Dict[str, str] = {}
_OCR_CACHE_FILE = BUILD_DIR / "ocr_cache.json"


# ──────────────────────────────────────────────
# Text / amount parsing helpers
# ──────────────────────────────────────────────
def parse_amount_from_text(text: str) -> Optional[float]:
    """Extracts the most plausible monetary amount from OCR'd receipt text.

    Prefers amounts labelled with specific financial keywords (balance, due,
    payable) over generic totals, and down-weights phone numbers, IDs, and
    implausibly large values. Returns None when nothing trustworthy is found.
    """
    if not text:
        return None

    best: Optional[float] = None
    best_score = float("-inf")
    for match in _AMOUNT_RE.finditer(text):
        context = match.group(1) or ""
        raw = match.group(2)
        if not raw:
            continue
        clean = raw.replace(",", "")
        try:
            value = float(clean)
        except ValueError:
            continue
        if value <= 0:
            continue

        score = 0.0
        if _MONEY_LABEL_SPECIFIC.search(context):
            score += 4.0
        elif _MONEY_LABEL_GENERAL.search(context):
            score += 3.0
        else:
            score -= 1.0
        if _PHONE_LABEL.search(context):
            score -= 6.0
        if "," in raw:
            score += 1.0
        if len(re.sub(r"\D", "", raw)) >= 10:
            score -= 3.0
        if re.fullmatch(r"(\d)\1{4,}", re.sub(r"\D", "", raw)):
            score -= 5.0
        if value > 1_000_000_000:
            score -= 4.0

        if score > best_score or (score == best_score and (best is None or value > best)):
            best = value
            best_score = score

    if best is None or best_score < 0:
        return None
    return best


def _load_ocr_cache() -> None:
    if _OCR_CACHE_FILE.exists():
        try:
            _OCR_TEXT_CACHE.update(json.loads(_OCR_CACHE_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError):
            pass


def _save_ocr_cache() -> None:
    try:
        _OCR_CACHE_FILE.write_text(json.dumps(_OCR_TEXT_CACHE), encoding="utf-8")
    except OSError:
        pass


# ──────────────────────────────────────────────
# OCR for blank event amounts
# ──────────────────────────────────────────────
def apply_ocr_amounts(
    events: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    usage_metrics_collector: List,
    request_date_str: Optional[str] = None,
) -> int:
    """Fills blank event amounts using recurrence evidence first, OCR second."""
    image_by_event = {img.get("related_event_id"): img for img in images if img.get("related_event_id")}

    from src.simulation.forecast import build_recurrence_patterns
    patterns = {}
    if request_date_str:
        for pat in build_recurrence_patterns(events, request_date_str):
            key = (pat["category"], pat["direction"], round(pat["interval_days"]))
            patterns.setdefault((pat["category"], pat["direction"]), pat["amount"])

    updated = 0
    _load_ocr_cache()
    for ev in events:
        if ev.get("amount") is not None:
            continue

        img = image_by_event.get(ev.get("event_id"))
        amount = None
        if img:
            # 1. Real receipt/evidence from the image overrides estimates.
            text = img.get("image_content") or ""
            if not text:
                image_id = img.get("image_id")
                text = _OCR_TEXT_CACHE.get(image_id, "")
                if not text:
                    image_path = DATASET_DIR / "media" / "images" / f"{image_id}.png"
                    if not image_path.exists():
                        continue
                    text, metrics = process_document(str(image_path))
                    usage_metrics_collector.append(metrics)
                    _OCR_TEXT_CACHE[image_id] = text
                img["image_content"] = text
            amount = parse_amount_from_text(text)

        # 2. Recurring category: reuse the established historical amount.
        if amount is None:
            amount = patterns.get((ev.get("category"), ev.get("direction")))
        if amount is not None:
            ev["amount"] = round(float(amount), 2)
            updated += 1
            logger.info(
                f"Backfilled amount {ev.get('amount')} for event {ev.get('event_id')}."
            )
    if updated:
        logger.info(f"Applied amounts to {updated} events.")
    _save_ocr_cache()
    return updated


# ──────────────────────────────────────────────
# Applying extracted facts to events
# ──────────────────────────────────────────────
def amount_appears_in_text(amount: Optional[float], text: str) -> bool:
    if amount is None or not text:
        return True
    normalized = re.sub(r"[,\s]", "", text)
    for candidate in (f"{amount:g}", f"{amount:.2f}", f"{amount:.1f}", str(int(amount))):
        if re.sub(r"[,\s]", "", candidate) in normalized:
            return True
    return False


def apply_facts_to_events(
    facts: List[Dict[str, Any]], events: List[Dict[str, Any]]
) -> int:
    """Applies extracted facts (amend/cancel/delay) to financial events."""
    events_by_id = {ev.get("event_id"): ev for ev in events if ev.get("event_id")}
    applied = 0
    for fact in facts:
        event_id = fact.get("event_id")
        if not event_id or event_id not in events_by_id:
            continue
        event = events_by_id[event_id]
        fact_type = fact.get("fact_type")
        source_text = fact.get("source_text") or ""

        if fact_type == "payment_failed":
            event["status"] = "cancelled"
            applied += 1
            logger.info(f"Cancelled event {event_id} because of a payment-failed message.")
            continue

        amount = fact.get("amount")
        if amount is None or not amount_appears_in_text(amount, source_text):
            continue
        if fact_type in ("salary_change", "rent_increase"):
            event["amount"] = float(amount)
            applied += 1
            logger.info(f"Amended amount of event {event_id} to {amount} from message evidence.")
    return applied


# ──────────────────────────────────────────────
# Spending changes
# ──────────────────────────────────────────────
def get_potential_spending_changes(
    events: List[Dict[str, Any]], user_profile: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identifies permitted, non-protected flexible spending changes."""
    willing_to_reduce = set((user_profile.get("expense_categories_user_is_willing_to_reduce") or "").split("|"))
    willing_to_stop = set((user_profile.get("expense_categories_user_is_willing_to_stop") or "").split("|"))
    protected = set((user_profile.get("expense_categories_to_protect") or "").split("|"))
    willing_to_reduce -= protected
    willing_to_stop -= protected

    changes = []
    for ev in events:
        category = ev.get("category")
        flexibility = ev.get("flexibility")
        status = ev.get("status")
        direction = ev.get("direction")

        if status not in ("settled", "scheduled") or direction != "debit":
            continue
        if not category:
            continue

        current_amount = float(ev.get("amount") or 0.0)
        if flexibility == "stoppable" and category in willing_to_stop:
            changes.append({
                "action": "stop",
                "event_id": ev["event_id"],
                "new_amount": 0.0,
                "savings": current_amount,
            })
        elif flexibility in ("reducible", "reducible_or_stoppable") and category in willing_to_reduce:
            min_amount = float(ev.get("minimum_allowed_amount") or 0.0)
            if current_amount > min_amount:
                changes.append({
                    "action": "reduce_to",
                    "event_id": ev["event_id"],
                    "new_amount": min_amount,
                    "savings": current_amount - min_amount,
                })
            elif flexibility == "reducible_or_stoppable" and category in willing_to_stop:
                changes.append({
                    "action": "stop",
                    "event_id": ev["event_id"],
                    "new_amount": 0.0,
                    "savings": current_amount,
                })

    changes.sort(key=lambda x: x["savings"], reverse=True)
    return changes[:3]


# ──────────────────────────────────────────────
# Payment schedules
# ──────────────────────────────────────────────
def parse_date(value: str, fallback: str) -> str:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return fallback


def generate_payment_schedule(option: Dict[str, Any], request_date_str: str) -> List[Dict[str, Any]]:
    """Builds the exact payment dates/amounts respecting the supplied option."""
    method = option.get("payment_method")
    amount = float(option.get("payment_amount") or 0.0)
    start = parse_date(option.get("first_payment_date") or request_date_str, request_date_str)

    if method == "full_payment":
        return [{"date": start, "amount": amount}]

    number_of_payments = max(1, int(option.get("number_of_payments") or 1))
    frequency_days = max(1, int(option.get("payment_frequency_days") or 30))
    schedule = []
    cursor = datetime.strptime(start, "%Y-%m-%d").date()
    for _ in range(number_of_payments):
        schedule.append({"date": cursor.strftime("%Y-%m-%d"), "amount": amount})
        cursor = cursor + timedelta(days=frequency_days)
    return schedule


# ──────────────────────────────────────────────
# Candidate plan generation
# ──────────────────────────────────────────────
def _find_viable(
    simulator: CashFlowSimulator,
    user_profile: Dict[str, Any],
    events: List[Dict[str, Any]],
    request_date_str: str,
    requested_amount: float,
    schedule: List[Dict[str, Any]],
    potential_changes: List[Dict[str, Any]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    if simulator.is_plan_viable(
        user_profile, events, request_date_str, requested_amount, schedule, []
    ):
        return True, []
    applied = []
    for change in potential_changes:
        applied.append(change)
        if simulator.is_plan_viable(
            user_profile, events, request_date_str, requested_amount, schedule, applied
        ):
            return True, applied
    return False, []


def build_candidates(
    simulator: CashFlowSimulator,
    user_profile: Dict[str, Any],
    events: List[Dict[str, Any]],
    request_date_str: str,
    requested_amount: float,
    desired_completion_date: str,
    allows_partial_payment: bool,
    payment_options: List[Dict[str, Any]],
    amount_safe_to_pay: float,
    earliest_date: Optional[str],
    potential_changes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    considered = set((user_profile.get("payment_methods_user_will_consider") or "").split("|"))
    max_months_raw = user_profile.get("max_installment_months")
    max_months = int(max_months_raw) if max_months_raw not in (None, "") else None

    candidates: List[Dict[str, Any]] = []

    def viable(schedule, changes) -> bool:
        return _find_viable(
            simulator, user_profile, events, request_date_str,
            requested_amount, schedule, potential_changes,
        )[0]

    for option in payment_options:
        method = option.get("payment_method")
        if method not in considered:
            continue

        schedule = generate_payment_schedule(option, request_date_str)
        if not schedule:
            continue

        if method == "full_payment":
            # A future-dated full payment is generated by the 'wait' candidate.
            if schedule[0]["date"] > request_date_str:
                continue
        elif method == "installments":
            number = int(option.get("number_of_payments") or 1)
            if max_months is not None and number > max_months:
                continue

        total_paid = float(option.get("total_payable_amount") or 0.0) or sum(
            p["amount"] for p in schedule
        )
        completion = schedule[-1]["date"]

        okay, changes = _find_viable(
            simulator, user_profile, events, request_date_str,
            requested_amount, schedule, potential_changes,
        )
        if okay:
            candidates.append({
                "payment_option_id": option.get("payment_option_id"),
                "method": method,
                "payments": schedule,
                "spending_changes": changes,
                "total_amount_paid": round(total_paid, 2),
                "completion_date": completion,
                "start_date": schedule[0]["date"],
            })
            continue

        # Even the cheapest payment option is not viable; try with spending changes.
        applied = []
        for change in potential_changes:
            applied.append(change)
            if simulator.is_plan_viable(
                user_profile, events, request_date_str, requested_amount, schedule, applied
            ):
                candidates.append({
                    "payment_option_id": option.get("payment_option_id"),
                    "method": method,
                    "payments": schedule,
                    "spending_changes": applied,
                    "total_amount_paid": round(total_paid, 2),
                    "completion_date": completion,
                    "start_date": schedule[0]["date"],
                })
                break

    # Partial payment: two exact payments derived from amount_safe_to_pay.
    if (
        allows_partial_payment
        and "partial_payment" in considered
        and 0.0 < amount_safe_to_pay < requested_amount
        and earliest_date
        and earliest_date <= desired_completion_date
    ):
        remainder = requested_amount - amount_safe_to_pay
        schedule = [
            {"date": request_date_str, "amount": amount_safe_to_pay},
            {"date": earliest_date, "amount": round(remainder, 2)},
        ]
        okay, changes = _find_viable(
            simulator, user_profile, events, request_date_str,
            requested_amount, schedule, potential_changes,
        )
        if okay:
            candidates.append({
                "payment_option_id": None,
                "method": "partial_payment",
                "payments": schedule,
                "spending_changes": changes,
                "total_amount_paid": round(requested_amount, 2),
                "completion_date": earliest_date,
                "start_date": request_date_str,
            })

    # Wait: full payment safe later (no spending changes by construction).
    if (
        "full_payment" in considered
        and earliest_date
        and earliest_date > request_date_str
        and earliest_date <= desired_completion_date
    ):
        schedule = [{"date": earliest_date, "amount": requested_amount}]
        candidates.append({
            "payment_option_id": None,
            "method": "wait",
            "payments": schedule,
            "spending_changes": [],
            "total_amount_paid": round(requested_amount, 2),
            "completion_date": earliest_date,
            "start_date": earliest_date,
        })

    # Baseline already covers the entire requested amount on the request date:
    # that full payment is affordable now by construction.
    if (
        "full_payment" in considered
        and amount_safe_to_pay >= requested_amount
        and not any(
            c["method"] == "full_payment" and c["start_date"] <= request_date_str
            for c in candidates
        )
    ):
        schedule = [{"date": request_date_str, "amount": round(requested_amount, 2)}]
        candidates.append({
            "payment_option_id": None,
            "method": "full_payment",
            "payments": schedule,
            "spending_changes": [],
            "total_amount_paid": round(requested_amount, 2),
            "completion_date": request_date_str,
            "start_date": request_date_str,
        })

    return candidates


# ──────────────────────────────────────────────
# Per-request processing
# ──────────────────────────────────────────────
def build_answer(
    best_plan: Optional[Dict[str, Any]],
    user_profile: Dict[str, Any],
    request_date_str: str,
    earliest_date: Optional[str],
    requested_amount: float,
) -> Dict[str, Any]:
    """Maps the best ranked plan to status/method/schedule/changes."""
    if best_plan is None:
        return {
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_schedule": [],
            "spending_changes": [],
            "earliest_date_for_full_payment": None,
        }

    method = best_plan["method"]
    schedule = best_plan["payments"]
    changes = best_plan["spending_changes"]
    start = best_plan["start_date"]

    if method == "wait":
        status = "affordable_later"
    elif method == "full_payment":
        if (
            not changes
            and schedule
            and schedule[0]["date"] <= request_date_str
            and earliest_date is not None
            and earliest_date <= request_date_str
        ):
            status = "affordable_now"
        elif changes:
            status = "affordable_with_plan"
        elif schedule and schedule[0]["date"] > request_date_str:
            status = "affordable_later"
        else:
            status = "affordable_with_plan"
    else:
        status = "affordable_with_plan"

    if status == "affordable_now":
        earliest_out = request_date_str
    elif status == "not_affordable":
        earliest_out = None
    else:
        earliest_out = earliest_date

    return {
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_schedule": schedule,
        "spending_changes": changes,
        "earliest_date_for_full_payment": earliest_out,
    }


def process_request(
    db_manager: AgentContextDB,
    simulator: CashFlowSimulator,
    ranker: PlanRanker,
    user_id: str,
    request_id: str,
    usage_metrics_collector: List,
) -> Dict[str, Any]:
    """Processes one request and returns the final formatted output row (minus request_id)."""
    logger.info(f"Processing request {request_id} for user {user_id}...")

    context = db_manager.get_user_request_context(user_id, request_id)
    user_profile = context["user_profile"]
    request = context["request"]
    events = context["events"]
    images = context["images"]
    messages = context["messages"]
    payment_options = context["payment_options"]

    request_date_str = (request.get("request_date") or "")[:10]
    requested_amount = float(request.get("requested_amount") or 0.0)
    desired_completion_date = (request.get("desired_completion_date") or "")[:10]
    allows_partial_payment = str(request.get("allows_partial_payment") or "").strip().lower() == "true"

    # 1. Backfill blank event amounts from linked images via OCR.
    apply_ocr_amounts(events, images, usage_metrics_collector, request_date_str)

    # 2. Extract and apply message/image facts (amend / cancel / delay).
    facts, extract_metrics = extract_facts_from_context({"messages": messages, "images": images})
    usage_metrics_collector.extend(extract_metrics)
    applied_count = apply_facts_to_events(facts, events)
    if applied_count:
        logger.info(f"Applied {applied_count} extracted facts for request {request_id}.")

    # 3. Baseline simulation (no optional spending changes).
    baseline = simulator.simulate(user_profile, events, request_date_str, requested_amount)
    amount_safe_to_pay = baseline["amount_safe_to_pay"]
    earliest_date = baseline["earliest_date_for_full_payment"]

    # 4. Generate and rank candidate plans.
    potential_changes = get_potential_spending_changes(events, user_profile)
    candidates = build_candidates(
        simulator, user_profile, events, request_date_str, requested_amount,
        desired_completion_date, allows_partial_payment, payment_options,
        amount_safe_to_pay, earliest_date, potential_changes,
    )
    best_plan = ranker.rank_plans(candidates, desired_completion_date)

    answer_parts = build_answer(
        best_plan, user_profile, request_date_str, earliest_date, requested_amount,
    )

    # 6. Explanation (LLM with deterministic fallback).
    explainer_data = {
        "status": answer_parts["affordability_status"],
        "method": answer_parts["recommended_payment_method"],
        "amount_safe_to_pay": amount_safe_to_pay,
        "earliest_date_for_full_payment": answer_parts["earliest_date_for_full_payment"],
        "payment_schedule": answer_parts["payment_schedule"],
        "spending_changes": answer_parts["spending_changes"],
        "user_home_currency": user_profile.get("home_currency", "USD"),
        "requested_amount": requested_amount,
        "request_date": request_date_str,
    }
    explanation, explain_metrics = generate_explanation(explainer_data)
    usage_metrics_collector.append(explain_metrics)

    # 7. Validate and serialize.
    answer = Answer(
        affordability_status=answer_parts["affordability_status"],
        recommended_payment_method=answer_parts["recommended_payment_method"],
        decision_explanation=explanation,
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date_for_full_payment=answer_parts["earliest_date_for_full_payment"],
        payment_schedule=[Payment(**p) for p in answer_parts["payment_schedule"]],
        spending_changes=[SpendingChange(**c) for c in answer_parts["spending_changes"]],
    )
    return validate_and_format_answer(answer)


def fallback_row(request_id: str, error: str) -> Dict[str, str]:
    """Guarantees a schema-compliant row when a request fails unexpectedly."""
    message = " ".join(str(error).split())
    logger.error(f"Fallback row for {request_id}: {message}")
    return {
        "request_id": request_id,
        "amount_safe_to_pay": 0.0,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": (
            f"The request could not be evaluated due to an internal error: {message[:200]}"
        ),
    }


def _insert_sample_requests(db_manager: AgentContextDB, requests_df: pd.DataFrame) -> None:
    """Inserts sample requests into the DB so sample users can be processed."""
    import sqlite3

    conn = sqlite3.connect(db_manager.db_path)
    rows = list(requests_df.to_dict("records"))
    conn.executemany(
        """
        INSERT OR REPLACE INTO requests
        (request_id, user_id, request_date, request_type, requested_amount,
         desired_completion_date, allows_partial_payment, request_text)
        VALUES
        (:request_id, :user_id, :request_date, :request_type, :requested_amount,
         :desired_completion_date, :allows_partial_payment, :request_text)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    logger.info(f"Upserted {len(rows)} sample requests into the database.")


def main():
    parser = argparse.ArgumentParser(description="Buy or Wait? AI Financial Agent")
    parser.add_argument("--user-id", type=str, help="Specific user ID to process")
    parser.add_argument("--request-id", type=str, help="Specific request ID to process")
    parser.add_argument(
        "--validate-samples",
        action="store_true",
        help="Run on dataset/sample_requests.csv and write sample_output.csv for comparison.",
    )
    args = parser.parse_args()

    # 1. Initialize database & components (rebuild from scratch each run).
    logger.info("Initializing database and components...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    db_manager = AgentContextDB(db_path=str(DB_PATH))
    db_manager.init()

    converter = CurrencyConverter(str(DB_PATH))
    simulator = CashFlowSimulator(converter)
    ranker = PlanRanker()

    # 2. Load the exact list of requests (1:1 output guarantee).
    if args.validate_samples:
        requests_df = pd.read_csv(DATASET_DIR / "sample_requests.csv", dtype=str)
        base_cols = [
            "request_id", "user_id", "request_date", "request_type",
            "requested_amount", "desired_completion_date",
            "allows_partial_payment", "request_text",
        ]
        requests_df = requests_df[base_cols]
        _insert_sample_requests(db_manager, requests_df)
        output_path = PROJECT_ROOT / "sample_output.csv"
        out_columns = OUTPUT_COLUMNS
    else:
        requests_df = pd.read_csv(DATASET_DIR / "requests.csv", dtype=str)
        output_path = OUTPUT_CSV
        out_columns = OUTPUT_COLUMNS

    if args.user_id and args.request_id:
        requests_df = requests_df[
            (requests_df["user_id"] == args.user_id) & (requests_df["request_id"] == args.request_id)
        ]
    elif args.request_id:
        requests_df = requests_df[requests_df["request_id"] == args.request_id]

    usage_metrics_collector: List = []
    results: List[Dict[str, Any]] = []

    # 3. Process every request; never skip a row.
    for _, row in requests_df.iterrows():
        user_id = row["user_id"]
        request_id = row["request_id"]
        try:
            result = process_request(
                db_manager, simulator, ranker, user_id, request_id, usage_metrics_collector
            )
            result["request_id"] = request_id
            results.append(result)
            logger.info(
                f"Processed {request_id}: {result['affordability_status']} / "
                f"{result['recommended_payment_method']}"
            )
        except ContextNotFoundError as exc:
            logger.error(f"Context not found for {request_id}: {exc}")
            results.append(fallback_row(request_id, exc))
        except Exception as exc:
            logger.exception(f"Failed to process {request_id}: {exc}")
            results.append(fallback_row(request_id, exc))

    # 4. Write schema-compliant output.csv to the project root.
    out_df = pd.DataFrame(results, columns=out_columns)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(out_df)} rows to {output_path}")

    # 5. Generate the evaluation usage report.
    try:
        from evaluation.main import generate_usage_report
        generate_usage_report(usage_metrics_collector, total_requests=len(out_df))
    except ImportError:
        logger.warning("evaluation.main not found. Skipping usage report generation.")
    except Exception as exc:
        logger.error(f"Failed to generate usage report: {exc}")


if __name__ == "__main__":
    main()