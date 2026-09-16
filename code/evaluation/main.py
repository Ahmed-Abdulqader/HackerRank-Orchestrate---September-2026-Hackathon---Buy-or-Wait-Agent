#!/usr/bin/env python3
"""
Evaluation Module - Usage Tracker
=================================
Aggregates LLM API usage metrics and generates a comprehensive 
Markdown report for evaluation and cost tracking.
"""

import sys
import logging
from pathlib import Path
from typing import List, Union
from collections import defaultdict

# Ensure the 'code' directory is in the Python path for absolute imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.schemas import UsageMetrics

logger = logging.getLogger(__name__)

# Resolve paths
EVAL_DIR = Path(__file__).resolve().parent
REPORT_PATH = EVAL_DIR / "usage_report.md"


def generate_usage_report(
    usage_metrics: List[Union[UsageMetrics, dict]], total_requests: int = 0
):
    """
    Aggregates usage metrics and writes a summary report to usage_report.md.

    Args:
        usage_metrics: A list of UsageMetrics objects or dictionaries containing
                       token usage and cost data from the LLM agents.
        total_requests: Number of requests processed (0 when unknown).
    """
    if not usage_metrics:
        logger.warning("No usage metrics provided. Generating empty report.")
        usage_metrics = []

    # Aggregate totals
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    # Group by endpoint and model
    endpoint_stats = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
    model_stats = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})

    for metric in usage_metrics:
        # Handle both Pydantic models and dictionaries
        if isinstance(metric, dict):
            m_input = metric.get("input_tokens", 0)
            m_output = metric.get("output_tokens", 0)
            m_cost = metric.get("cost", 0.0)
            m_endpoint = metric.get("endpoint", "unknown")
            m_model = metric.get("model_name", "unknown")
        else:
            m_input = metric.input_tokens
            m_output = metric.output_tokens
            m_cost = metric.cost
            m_endpoint = metric.endpoint or "unknown"
            m_model = metric.model_name or "unknown"

        total_input_tokens += m_input
        total_output_tokens += m_output
        total_cost += m_cost

        endpoint_stats[m_endpoint]["calls"] += 1
        endpoint_stats[m_endpoint]["input_tokens"] += m_input
        endpoint_stats[m_endpoint]["output_tokens"] += m_output
        endpoint_stats[m_endpoint]["cost"] += m_cost

        model_stats[m_model]["calls"] += 1
        model_stats[m_model]["input_tokens"] += m_input
        model_stats[m_model]["output_tokens"] += m_output
        model_stats[m_model]["cost"] += m_cost

    # Dynamically extract model names for the report header
    detected_models = ", ".join(sorted(model_stats.keys())) if model_stats else "None"

    # Generate Markdown Content
    avg_input = total_input_tokens / total_requests if total_requests else 0.0
    avg_output = total_output_tokens / total_requests if total_requests else 0.0
    avg_cost = total_cost / total_requests if total_requests else 0.0

    md_lines = [
        "# AI Agent Usage Report",
        "",
        "## Summary",
        f"- **Active Models**: {detected_models}",
        f"- **Requests Processed**: {total_requests}",
        f"- **Total API Calls**: {len(usage_metrics)}",
        f"- **Total Input Tokens**: {total_input_tokens:,}",
        f"- **Total Output Tokens**: {total_output_tokens:,}",
        f"- **Total Estimated Cost**: ${total_cost:.4f}",
        f"- **Average Input Tokens / Request**: {avg_input:,.2f}",
        f"- **Average Output Tokens / Request**: {avg_output:,.2f}",
        f"- **Average Cost / Request**: ${avg_cost:.6f}",
        "",
        "## Breakdown by Endpoint",
        "",
        "| Endpoint | Calls | Input Tokens | Output Tokens | Cost ($) |",
        "|---|---|---|---|---|",
    ]

    for endpoint, stats in sorted(endpoint_stats.items()):
        md_lines.append(
            f"| {endpoint} | {stats['calls']} | {stats['input_tokens']:,} | "
            f"{stats['output_tokens']:,} | {stats['cost']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Breakdown by Model",
        "",
        "| Model | Calls | Input Tokens | Output Tokens | Cost ($) |",
        "|---|---|---|---|---|",
    ])

    for model, stats in sorted(model_stats.items()):
        md_lines.append(
            f"| {model} | {stats['calls']} | {stats['input_tokens']:,} | "
            f"{stats['output_tokens']:,} | {stats['cost']:.4f} |"
        )

    md_content = "\n".join(md_lines)

    # Write to file
    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"✅ Usage report successfully written to {REPORT_PATH}")
    except Exception as e:
        logger.error(f"Failed to write usage report: {e}")


if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Updated dummy metrics matching actual codebase configurations
    dummy_metrics = [
        UsageMetrics(model_name="gemini-2.5-flash", input_tokens=1500, output_tokens=250, cost=0.001075, endpoint="image_handler_vision"),
        UsageMetrics(model_name="rapidocr_onnxruntime", input_tokens=0, output_tokens=0, cost=0.0, endpoint="image_handler_local"),
        UsageMetrics(model_name="gemini-3.5-flash", input_tokens=2500, output_tokens=450, cost=0.0078, endpoint="extractor"),
        UsageMetrics(model_name="gemini-3.5-flash", input_tokens=1800, output_tokens=600, cost=0.0081, endpoint="explainer"),
        UsageMetrics(model_name="gemini-3.5-flash", input_tokens=500, output_tokens=100, cost=0.00165, endpoint="validator"),
    ]
    
    generate_usage_report(dummy_metrics)
    print(f"Test report generated at {REPORT_PATH}")