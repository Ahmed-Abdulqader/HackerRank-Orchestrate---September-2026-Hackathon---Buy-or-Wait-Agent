# "Buy or Wait?" AI Financial Agent

**HackerRank Orchestrate — September 2026 Hackathon**

An end-to-end financial decision engine that reads structured profiles and financial
events, forecasts recurring cash flow, parses OCR'd receipts, and — for every purchase
or payment request — recommends a deterministic, affordable payment action:
`affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`.

The engine treats unstructured messages and images as untrusted evidence: they may
clarify, amend, delay, cancel, or confirm a financial fact, but their embedded
instructions never override the challenge rules.

---

## 1. Overview

For each row in `dataset/requests.csv`, the agent answers: *can this request be paid
safely, and how?*

The answer must be grounded in:

- the user's current available balance and preferred minimum balance to keep;
- historical, pending, scheduled, and settled transactions (including foreign-currency
  cash events converted at fixed dated exchange rates);
- recurring commitments (rent, utilities, groceries, salary) detected from history and
  projected forward;
- the seller/provider payment options available for the request plus the user's
  payment preferences;
- relevant support from messages and OCR'd images (receipts, letters, statements);
- protected categories and willingness to stop or reduce flexible spending.

The system produces one output row per request with a payment plan, an earliest safe
full-payment date, optional spending changes, and a grounded explanation.

---

## 2. Core Architecture & System Design

The pipeline is split into two clean layers:

### 2.1 Deterministic Simulation Engine (`src/simulation/`)

Every numeric recommendation is produced by **pure, deterministic math** — no LLM is
involved in computing balances, safe amounts, or payment plans. LLMs only generate
prose *after* the decision is already fixed, and they are instructed never to alter,
recompute, or contradict the computed values.

Key modules:

| Module | Responsibility |
|---|---|
| `simulator.py` | Cash-flow simulation over a 90-day horizon (extendable for long plans). Converts foreign-currency events, builds a daily balance curve, computes `amount_safe_to_pay`, the `earliest_date_for_full_payment`, and plan viability. |
| `forecast.py` | Recurrence engine: detects recurring debit/credit patterns from settled history and projects them forward (windowed, with cadence and spread guards). |
| `ranker.py` | Deterministic plan ranking: prefers plans that finish by the deadline, avoid spending changes, minimize total cost, start earlier, and use fewer payments. |
| `currency.py` | Fixed, dated exchange-rate conversion using `dataset/exchange_rates.csv`. |

#### Next-pay-cycle cutoff

`amount_safe_to_pay` (the amount safe *today*) is judged against the **next pay
cycle**: commitments are projected only until the next confirmed/recurring salary
receipt, mirroring the conservative "how much will remain before my next paycheck"
decision. `earliest_date_for_full_payment` is computed over the full forecast window.

### 2.2 Recurring Expense Forecasting (`forecast.py`)

Historical transactions are grouped by `(category, direction)`. A category is
considered recurring when:

- it has at least **3 settled occurrences** in the previous **400 days**;
- the pairwise gap between consecutive occurrences is stable — median gap within
  **5–40 days** and median absolute deviation bounded by `max(3, 0.35 × median)`;
- the projected amount is the **median of the last 3 settled amounts**, which is
  resistant to one-off spikes.

Projected occurrences start from the last historical date and step forward by the
detected cadence. Dates already covered by a real in-horizon event are skipped, and
category-level spending changes (`stop:<id>` / `reduce_to:<id>:<amount>`) cap or
remove the corresponding projected flows. Only non-protected, flexible categories the
user permits may be adjusted.

> Note: a classic bug fixed during development was pairing `zip(dates, sorted(dates))`,
> which pairs every date with *itself* (all-zero gaps). The correct adjacency pairing is
> `zip(sorted_dates, sorted_dates[1:])`.

### 2.3 OCR & Fact Extraction (`image_handler.py`)

Blank event amounts are backfilled in priority order:

1. **Image receipt first** — when an event has a `related_event_id` in `images.csv`,
   the linked receipt is the ground truth.
2. **Recurrence estimate second** — if no image exists, the historical pattern amount
   is used as a fallback.

The local OCR path uses **RapidOCR** (ONNX Runtime). A label-scoring amount parser then
selects the correct figure from the raw text. The parser was built to handle
**Indian-style currency formats** (`2,00,000.00`, `1,00,000.00`) while suppressing
classic false positives:

- **+4** for specific money labels: `balance`, `due`, `payable`, `outstanding`,
  `received`, `paid`, `refund`;
- **+3** for general money labels: `total`, `amount`, `bill`, `charge`, `fee`, `rent`,
  `sum`, `price`, `cost`, `invoice`;
- **−6** for phone-like context: `phone`, `tel`, `contact`, `mobile`, `reg`, `uid`,
  `patient`, `admission`, `address`;
- penalties for ≥10-digit numbers, long repeated-digit runs, or amounts > 1e9.

This prevents picking up 10-digit phone numbers (`9403265989`) or contact IDs
(`0006666666`) that routinely appear on invoices and hospital bills. OCR results are
cached to `build/ocr_cache.json` so re-runs skip repeated image processing.

### 2.4 Fault-Tolerant LLM Pipeline (`src/agents/`)

{% raw %}

| Module | Role | Failure mode |
|---|---|---|
| `extractor.py` | Structured fact extraction from messages/images (salary change, rent increase, refund, etc.) via a **Google Gemini flash** model | Any error → `0` facts returned, pipeline continues |
| `explainer.py` | Writes the `decision_explanation` prose from pre-computed values | Any error → deterministic, figure-rich fallback text |
| `image_handler.py` | Local RapidOCR + guarded vision-LLM fallback | OCR/vision errors → returns OCR text as-is, never crashes the run |

{% endraw %}

The agents use a **zero-retry** `try/except` wrapper (and `retries=0`): on a network
error, 401/402, or 429 quota rejection (free-tier limits are easy to exhaust in a 250-row
run), the agent **immediately** catches the exception and returns the deterministic
fallback. No retries, no hangs, every request still completes. Env-driven keys come from
`.env` (`GOOGLE_API_KEY`); the configured model is the lightweight `gemini-3.5-flash`
(the `gemini-1.5-flash` alias is no longer served on the v1beta endpoint). The intended
flash tier is overridable via a single constant at the top of each agent module.

---

## 3. Pipeline Lifecycle & Directory Structure

```text
code/
├── main.py                        # Orchestrator: build DB -> iterate requests -> write output.csv
├── README.md                      # This document
└── src/
    ├── database/
    │   ├── build_context.py       # SQLite ingestion: dataset/ -> build/agent_context.db
    │   └── sanitize.py            # Strict NULL/type rules during CSV parsing
    ├── simulation/
    │   ├── currency.py            # Fixed dated exchange-rate conversion
    │   ├── forecast.py            # Recurrence detection + forward projection
    │   ├── simulator.py           # Cash-flow engine, safe-amount, plan viability
    │   └── ranker.py              # Deterministic plan ranking
    ├── agents/
    │   ├── extractor.py           # Gemini fact extraction (structured) + fallback
    │   ├── explainer.py           # Gemini prose generation + deterministic fallback
    │   └── image_handler.py       # RapidOCR + label-scoring amount parser
    ├── models/
    │   └── schemas.py             # Pydantic output/status/method enums & validation
    └── utils/
        ├── costs.py               # Per-model token-cost estimation
        └── validator.py           # Strict schema validator + pipe-delimited formatter
```

### Execution flow per request

1. **Context load** — user profile, request, events, payment options, messages, images
   are pulled from `build/agent_context.db`.
2. **OCR backfill** — blank event amounts are filled from linked receipt images
   (recurrence amount as fallback).
3. **Fact extraction & application** — messages/images may amend, cancel, or confirm
   events (e.g., a rent-increase letter supersedes the historical rent figure).
4. **Baseline simulation** — daily balance curve over the next pay cycle / 90-day
   window; `amount_safe_to_pay` and `earliest_date_for_full_payment` computed.
5. **Candidate generation** — full payment, installments, partial payment (two legs),
   and wait plans are built from the supplied payment options and the baseline safety
   envelope; spending changes up to three flexible reductions are tried when needed.
6. **Ranking** — `ranker.py` selects the best viable plan by the tie-break ladder.
7. **Status mapping** — a plan maps to one of the four `affordability_status` values
   with the correct method, schedule, and earliest date.
8. **Explanation** — Gemini (or the deterministic fallback) writes a grounded,
   figure-rich `decision_explanation`.
9. **Validation** — every row is format-checked and normalized before being written.

---

## 4. Setup & How to Run

### Prerequisites

- Python 3.10+ (development used CPython 3.14)
- A virtual environment with the project dependencies installed:

```bash
cd <repo-root>/code
uv sync 
```



- **`.env` configuration** at the repository root:

```bash
# Optional runtime keys. If absent, agents produce deterministic fallbacks only.
GOOGLE_API_KEY=your_key_here
# OPENROUTER_API_KEY=optional_legacy_key
```

No API key is strictly required to produce a valid `output.csv` — every LLM call has a
deterministic fallback, so the full pipeline runs offline if the key is missing.

### Repo layout the code expects

```text
<repo-root>/
├── dataset/               # read-only inputs (requests, events, options, media/)
├── build/                 # runtime artifacts (SQLite DB, OCR cache) — generated
├── code/                  # this solution
└── output.csv             # generated predictions (repository root)
```

### Execution commands

Process **all 250 evaluation requests** and write the root `output.csv`:

```bash
cd code && PYDANTIC_AI_NO_BANNER=1 ../.venv/bin/python main.py
```

**Sample ground-truth validation mode** — processes `dataset/sample_requests.csv` and
writes `sample_output.csv` at the repo root (useful for tuning against the 25 labeled
examples):

```bash
cd code && PYDANTIC_AI_NO_BANNER=1 ../.venv/bin/python main.py --validate-samples
```

**Process a single user or request** (debugging):

```bash
cd code && PYDANTIC_AI_NO_BANNER=1 ../.venv/bin/python main.py --user-id user_03
cd code && PYDANTIC_AI_NO_BANNER=1 ../.venv/bin/python main.py --request-id request_16
```

All flags:

| Flag | Behavior |
|---|---|
| `--user-id <id>` | Only process the given user's requests |
| `--request-id <id>` | Only process a single request |
| `--validate-samples` | Load `dataset/sample_requests.csv` and write `sample_output.csv` |

`PYDANTIC_AI_NO_BANNER=1` simply silences the pydantic-ai startup banner.

---

## 5. Data Compliance & Output Specification

### Strict read-only guarantee

- Input CSVs under `dataset/` are **never modified** — the solution only reads them.
- All runtime artifacts live under the **`build/` directory**: `build/agent_context.db`
  (SQLite ingestion) and `build/ocr_cache.json` (OCR text cache).
- Organizer-only files outside `dataset/` are never referenced for predictions.
- No hardcoded labels for the 250 evaluation requests; the engine is fully general.

### Output header & schema

`output.csv` (repository root) has exactly these columns in this order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

### Enumerations

| Column | Allowed values |
|---|---|
| `affordability_status` | `affordable_now`, `affordable_with_plan`, `affordable_later`, `not_affordable` |
| `recommended_payment_method` | `full_payment`, `partial_payment`, `installments`, `wait`, `not_recommended` |

### Formatting rules

- `amount_safe_to_pay` is `0 <= amount_safe <= requested_amount`, formatted without
  trailing `.0` (whole numbers like `25256`, fractional like `620.40`).
- `payment_plan` is `none` or chronological `YYYY-MM-DD:amount` entries joined by `|`
  (e.g., `2025-08-08:15952906.67|2025-09-07:15952906.67|2025-10-07:15952906.67`).
  - `partial_payment` uses exactly two entries: `amount_safe_to_pay` on
    `request_date`, then the remainder on `earliest_date_for_full_payment` (the sum
    equals `requested_amount`).
  - `installments` exactly follows a supplied payment option (dates and amounts).
- `spending_changes_needed` is `none` or up to three pipe-delimited
  `stop:<event_id>` / `reduce_to:<event_id>:<amount>` actions, targeting only
  non-protected, flexible recurring events.
- `earliest_date_for_full_payment` is `request_date` for `affordable_now`, empty for
  `not_affordable`, and a date otherwise.
- `decision_explanation` is a concise, grounded explanation citing concrete amounts
  and dates.

### Validation

`src/utils/validator.py` enforces every rule above before writing a row; a malformed
row cannot silently reach the output file.

---

## Deliverable Summary

| Artifact | Location |
|---|---|
| Predictions (250 rows) | `<repo-root>/output.csv` |
| Sample validation output | `<repo-root>/sample_output.csv` (only from `--validate-samples`) |
| Runnable solution | `<repo-root>/code/` |
| Runtime DB / cache | `<repo-root>/build/` |
| Chat transcript | `<repo-root>/log.txt` |
