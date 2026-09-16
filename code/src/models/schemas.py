from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Payment(BaseModel):
    """Represents a single scheduled payment in the payment plan."""
    date: str = Field(..., description="ISO 8601 date string (YYYY-MM-DD) of the payment.")
    amount: float = Field(..., description="The amount to be paid on this date.")


class SpendingChange(BaseModel):
    """Represents a required change to a user's recurring expense to make a plan viable."""
    action: Literal["stop", "reduce_to"] = Field(
        ..., description="The action to take: 'stop' the expense entirely, or 'reduce_to' a specific amount."
    )
    event_id: str = Field(..., description="The ID of the financial event to modify.")
    new_amount: Optional[float] = Field(
        default=None, description="The new amount for the expense. Required if action is 'reduce_to', ignored if 'stop'."
    )


class Answer(BaseModel):
    """
    The final structured output of the AI financial decision agent.
    All numerical and date computations must be done in pure Python before
    populating this model. The LLM only generates the decision_explanation.
    """
    affordability_status: Literal[
        "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"
    ] = Field(
        ...,
        description=(
            "Whether the request is affordable now, affordable with a plan, "
            "affordable later, or not affordable."
        ),
    )
    recommended_payment_method: Literal[
        "full_payment", "partial_payment", "installments", "wait", "not_recommended"
    ] = Field(
        ...,
        description="The recommended payment approach: pay in full, partially, in installments, wait, or not proceed.",
    )
    decision_explanation: str = Field(
        ...,
        description=(
            "A concise, grounded explanation of the recommendation using concrete "
            "figures, dates, and currency amounts."
        ),
    )
    amount_safe_to_pay: float = Field(
        ...,
        description="The maximum amount that can be safely paid on the request date before optional spending changes.",
    )
    earliest_date_for_full_payment: Optional[str] = Field(
        default=None,
        description="Earliest date (YYYY-MM-DD) when the full requested amount is safe as one payment.",
    )
    payment_schedule: List[Payment] = Field(
        default_factory=list,
        description="Chronological scheduled payments used to build the payment_plan string.",
    )
    spending_changes: List[SpendingChange] = Field(
        default_factory=list,
        description="Recommended spending changes to make the chosen plan viable.",
    )


class UsageMetrics(BaseModel):
    """Tracks token usage and cost for a single LLM API call for evaluation reporting."""
    model_name: str = Field(..., description="The name of the LLM model used.")
    input_tokens: int = Field(..., description="Number of prompt/input tokens.")
    output_tokens: int = Field(..., description="Number of completion/output tokens.")
    cost: float = Field(..., description="Estimated cost of this API call in USD.")
    endpoint: Optional[str] = Field(default=None, description="The specific endpoint or agent name (e.g., 'explainer', 'image_handler').")