import os
import logging
from typing import List, Dict, Any, Optional, Literal, Tuple
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
# Pydantic Models for Structured Extraction
# ──────────────────────────────────────────────
class ExtractedFact(BaseModel):
    """Represents a single structured financial fact extracted from unstructured text."""
    fact_type: Literal[
        "salary_change", "invoice_approved", "rent_increase",
        "refund_processing", "bonus_pending", "payment_failed", "other"
    ] = Field(..., description="The category of the extracted financial fact.")
    description: str = Field(..., description="A brief, clear summary of the fact.")
    amount: Optional[float] = Field(None, description="The monetary amount mentioned, if any.")
    currency: Optional[str] = Field(None, description="The currency of the amount (e.g., USD, EUR, IDR, ZAR).")
    date: Optional[str] = Field(None, description="The relevant date in YYYY-MM-DD format, if explicitly mentioned.")


class ExtractionResult(BaseModel):
    """Wrapper for the LLM's structured output."""
    facts: List[ExtractedFact] = Field(default_factory=list, description="List of extracted financial facts.")


# ──────────────────────────────────────────────
# Pydantic AI Agent Initialization
# ──────────────────────────────────────────────
_EXTRACTOR_MODEL = "gemini-3.5-flash"
_api_key = os.getenv('GOOGLE_API_KEY')
if _api_key:
    _extractor_agent = Agent(
        model=GoogleModel(_EXTRACTOR_MODEL, provider=GoogleProvider(api_key=_api_key)),
        output_type=ExtractionResult,
        retries=0,
        system_prompt=(
            "You are an expert financial fact extractor. "
            "Analyze the provided text (from user messages or OCR'd images) and extract concrete financial facts. "
            "Focus specifically on: salary changes, approved invoices, rent increases, pending refunds, "
            "bonus updates, or failed payments. "
            "Extract the exact amount, currency, and date only when they are explicitly stated in the text. "
            "If the text contains no concrete financial facts, return an empty list. "
            "Do not hallucinate numbers, dates, or amounts."
        ),
    )
else:
    _extractor_agent = None
    logger.warning("GOOGLE_API_KEY is not set. Fact extraction will be skipped.")


# ──────────────────────────────────────────────
# Extraction Functions
# ──────────────────────────────────────────────
def _empty_metrics(endpoint: str) -> UsageMetrics:
    return UsageMetrics(
        model_name="none", input_tokens=0, output_tokens=0, cost=0.0, endpoint=endpoint
    )


def extract_facts_from_text(text: str, source_id: str) -> Tuple[ExtractionResult, UsageMetrics]:
    """Extracts structured facts from a single text string using the LLM agent."""
    if not text or not text.strip():
        return ExtractionResult(facts=[]), _empty_metrics("extractor")

    if _extractor_agent is None:
        return ExtractionResult(facts=[]), _empty_metrics("extractor_skipped")

    prompt = f"Source ID: {source_id}\n\nText to analyze:\n{text}"
    try:
        response = _extractor_agent.run_sync(prompt)
        usage = response.usage
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        metrics = UsageMetrics(
            model_name=_EXTRACTOR_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=estimate_cost(_EXTRACTOR_MODEL, input_tokens, output_tokens),
            endpoint="extractor",
        )
        logger.info(f"Extracted {len(response.output.facts)} facts from {source_id}.")
        return response.output, metrics
    except Exception as exc:
        logger.error(f"Extraction agent failed for {source_id}: {exc}")
        return ExtractionResult(facts=[]), UsageMetrics(
            model_name=_EXTRACTOR_MODEL, input_tokens=0, output_tokens=0, cost=0.0,
            endpoint="extractor_error",
        )


def extract_facts_from_context(context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[UsageMetrics]]:
    """
    Extracts facts from all messages and image contents in the user's context.

    Each returned fact dict is enriched with the `event_id` and source message
    text of the message/image it came from, so downstream code can apply the
    fact to the matching financial-event row.
    """
    all_facts: List[Dict[str, Any]] = []
    all_metrics: List[UsageMetrics] = []

    messages = context.get("messages", [])
    for msg in messages:
        content = msg.get("message_text") or msg.get("content") or ""
        text = content.strip()
        msg_id = msg.get("message_id", "unknown")
        event_id = msg.get("related_event_id")
        if text:
            results, metrics = extract_facts_from_text(text, f"message_{msg_id}")
            all_metrics.append(metrics)
            for fact in results.facts:
                fact_dict = fact.model_dump()
                fact_dict["event_id"] = event_id or None
                fact_dict["source_text"] = text
                all_facts.append(fact_dict)

    images = context.get("images", context.get("related_images", []))
    for img in images:
        content = img.get("image_content") or ""
        text = content.strip()
        img_id = img.get("image_id", "unknown")
        event_id = img.get("related_event_id")
        if text:
            results, metrics = extract_facts_from_text(text, f"image_{img_id}")
            all_metrics.append(metrics)
            for fact in results.facts:
                fact_dict = fact.model_dump()
                fact_dict["event_id"] = event_id or None
                fact_dict["source_text"] = text
                all_facts.append(fact_dict)

    logger.info(f"Total facts extracted from context: {len(all_facts)}")
    return all_facts, all_metrics