import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from .sanitize import sanitize_text

logger = logging.getLogger(__name__)


class ContextNotFoundError(ValueError):
    """Raised when a requested user_id or request_id does not exist."""
    pass


# Resolve project root (3 levels up from this file: database -> src -> code -> root)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if not (PROJECT_ROOT / "dataset").exists():
        raise FileNotFoundError
except Exception:
    PROJECT_ROOT = Path.cwd()

DATASET_DIR = str(PROJECT_ROOT / "dataset")
BUILD_DIR = str(PROJECT_ROOT / "build")
DEFAULT_DB_PATH = str(Path(BUILD_DIR) / "agent_context.db")


class AgentContextDB:
    """Manages schema creation, full reload from CSV, and context retrieval.

    The database lives in <project_root>/build and is dropped and rebuilt on
    every `init()` call so stale rows never persist between runs. The
    `dataset/` directory is never written to.
    """

    def __init__(self, dataset_dir: str = DATASET_DIR, db_path: str = DEFAULT_DB_PATH):
        self.dataset_dir = Path(dataset_dir)
        self.db_path = Path(db_path)

    def init(self):
        """Drops and rebuilds all tables, then reloads every CSV."""
        logger.info(f"Initializing database at {self.db_path} ...")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF;")
            self._drop_schema(conn)
            self._create_schema(conn)
            self._load_all_data(conn)
            conn.execute("PRAGMA foreign_keys = ON;")
            logger.info("Database schema and data load complete.")
        finally:
            conn.close()

    def _drop_schema(self, conn: sqlite3.Connection):
        for table in ("financial_profiles", "requests", "request_payment_options",
                      "financial_events", "messages", "images", "exchange_rates"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute("DROP VIEW IF EXISTS vw_user_active_profile")
        conn.commit()

    def _create_schema(self, conn: sqlite3.Connection):
        """DDL matching the ACTUAL CSV headers in dataset/."""
        ddl_statements = [
            """CREATE TABLE financial_profiles (
                user_id TEXT PRIMARY KEY, home_currency TEXT NOT NULL,
                current_available_balance REAL, minimum_balance_to_keep REAL,
                financial_priorities TEXT, expense_categories_to_protect TEXT,
                expense_categories_user_is_willing_to_reduce TEXT,
                expense_categories_user_is_willing_to_stop TEXT,
                payment_methods_user_will_consider TEXT,
                max_installment_months INTEGER
            );""",
            """CREATE TABLE requests (
                request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                request_date TEXT, request_type TEXT, requested_amount REAL,
                desired_completion_date TEXT, allows_partial_payment TEXT,
                request_text TEXT,
                FOREIGN KEY (user_id) REFERENCES financial_profiles(user_id)
            );""",
            """CREATE TABLE request_payment_options (
                payment_option_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                payment_method TEXT, payment_amount REAL, number_of_payments INTEGER,
                first_payment_date TEXT, payment_frequency_days INTEGER,
                financing_fee REAL, total_payable_amount REAL,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            );""",
            """CREATE TABLE financial_events (
                event_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                event_type TEXT, description TEXT, category TEXT, direction TEXT,
                amount REAL, currency TEXT, event_date TEXT, settlement_date TEXT,
                status TEXT, linked_event_id TEXT, flexibility TEXT, minimum_allowed_amount REAL,
                FOREIGN KEY (user_id) REFERENCES financial_profiles(user_id)
            );""",
            """CREATE TABLE messages (
                message_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                request_id TEXT, related_event_id TEXT, sent_at TEXT,
                source_type TEXT, message_text TEXT,
                FOREIGN KEY (user_id) REFERENCES financial_profiles(user_id),
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            );""",
            """CREATE TABLE images (
                image_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                request_id TEXT, related_event_id TEXT, image_content TEXT,
                FOREIGN KEY (user_id) REFERENCES financial_profiles(user_id),
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            );""",
            """CREATE TABLE exchange_rates (
                rate_date TEXT, from_currency TEXT, to_currency TEXT, rate REAL,
                UNIQUE(rate_date, from_currency, to_currency)
            );""",
            """CREATE VIEW vw_user_active_profile AS
            SELECT
                fp.user_id, fp.home_currency, fp.current_available_balance,
                fp.minimum_balance_to_keep, fp.financial_priorities,
                fp.expense_categories_to_protect,
                fp.expense_categories_user_is_willing_to_reduce,
                fp.expense_categories_user_is_willing_to_stop,
                fp.payment_methods_user_will_consider, fp.max_installment_months
            FROM financial_profiles fp;""",
        ]

        cur = conn.cursor()
        for ddl in ddl_statements:
            cur.execute(ddl)
        conn.commit()

    def _load_csv(self, conn: sqlite3.Connection, csv_name: str, table_name: str,
                  columns: List[str], text_columns: List[str]):
        """Reads CSV, sanitizes text fields, and inserts all rows."""
        csv_path = self.dataset_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path, dtype=str)
        except pd.errors.EmptyDataError:
            return

        if df.empty:
            return

        present_cols = [c for c in columns if c in df.columns]
        df = df[present_cols]

        numeric_cols = {
            "current_available_balance", "minimum_balance_to_keep", "amount",
            "minimum_allowed_amount", "rate", "max_installment_months",
            "requested_amount", "payment_amount", "number_of_payments",
            "payment_frequency_days", "financing_fee", "total_payable_amount",
        }
        for col in present_cols:
            if col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in present_cols:
            if col in text_columns:
                df[col] = df[col].map(sanitize_text)

        df = df.where(pd.notna(df), None)
        rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

        placeholders = ", ".join(["?"] * len(present_cols))
        col_names = ", ".join(present_cols)
        sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"

        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
        logger.info(f"Loaded {len(rows)} rows into {table_name}.")

    def _populate_image_content(self, conn: sqlite3.Connection):
        """Best-effort local OCR to fill image_content for caching in the DB.

        OCR is intentionally lazy here: image_content is populated on-demand in
        code/main.py (apply_ocr_amounts) only for images linked to blank-amount
        events of the user being processed, so a full rebuild stays fast.
        """
        return

    def _load_all_data(self, conn: sqlite3.Connection):
        """Orchestrates the loading of all CSV files and image OCR."""
        self._load_csv(conn, "financial_profiles.csv", "financial_profiles",
            ["user_id", "home_currency", "current_available_balance", "minimum_balance_to_keep",
             "financial_priorities", "expense_categories_to_protect",
             "expense_categories_user_is_willing_to_reduce", "expense_categories_user_is_willing_to_stop",
             "payment_methods_user_will_consider", "max_installment_months"],
            text_columns=[])

        self._load_csv(conn, "requests.csv", "requests",
            ["request_id", "user_id", "request_date", "request_type", "requested_amount",
             "desired_completion_date", "allows_partial_payment", "request_text"],
            text_columns=["request_text"])

        self._load_csv(conn, "request_payment_options.csv", "request_payment_options",
            ["payment_option_id", "request_id", "payment_method", "payment_amount",
             "number_of_payments", "first_payment_date", "payment_frequency_days",
             "financing_fee", "total_payable_amount"],
            text_columns=[])

        self._load_csv(conn, "financial_events.csv", "financial_events",
            ["event_id", "user_id", "event_type", "description", "category", "direction",
             "amount", "currency", "event_date", "settlement_date", "status",
             "linked_event_id", "flexibility", "minimum_allowed_amount"],
            text_columns=["description"])

        self._load_csv(conn, "messages.csv", "messages",
            ["message_id", "user_id", "request_id", "related_event_id", "sent_at",
             "source_type", "message_text"],
            text_columns=["message_text"])

        self._load_csv(conn, "images.csv", "images",
            ["image_id", "user_id", "request_id", "related_event_id"],
            text_columns=[])

        self._load_csv(conn, "exchange_rates.csv", "exchange_rates",
            ["rate_date", "from_currency", "to_currency", "rate"],
            text_columns=[])

        self._populate_image_content(conn)

    def get_user_request_context(self, user_id: str, request_id: str) -> Dict[str, Any]:
        """
        Retrieves the user's active profile, the request row, every financial
        event for the user, plus supporting messages, images, and payment options.

        Raises ContextNotFoundError if the user or request does not exist to
        prevent hallucinations.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute("SELECT * FROM vw_user_active_profile WHERE user_id = ?", (user_id,))
            profile_row = cur.fetchone()
            if not profile_row:
                raise ContextNotFoundError(
                    f"user_id '{user_id}' not found. Refusing to hallucinate.")

            cur.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,))
            request_row = cur.fetchone()
            if not request_row:
                raise ContextNotFoundError(
                    f"request_id '{request_id}' not found. Refusing to hallucinate.")

            cur.execute(
                "SELECT * FROM financial_events WHERE user_id = ? ORDER BY "
                "COALESCE(settlement_date, event_date)", (user_id,))
            events = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT * FROM messages WHERE user_id = ? OR request_id = ? "
                "ORDER BY sent_at", (user_id, request_id))
            messages = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT * FROM images WHERE user_id = ? OR request_id = ?",
                (user_id, request_id))
            images = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT * FROM request_payment_options WHERE request_id = ?",
                (request_id,))
            payment_options = [dict(r) for r in cur.fetchall()]

            return {
                "user_profile": dict(profile_row),
                "request": dict(request_row),
                "events": events,
                "messages": messages,
                "images": images,
                "payment_options": payment_options,
            }

        except ContextNotFoundError:
            raise
        except sqlite3.Error as exc:
            logger.error(f"SQLite error: {exc}")
            raise
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    db_manager = AgentContextDB()
    db_manager.init()
    logger.info("Running smoke test ...")
    try:
        ctx = db_manager.get_user_request_context("user_26", "request_26")
        logger.info("Smoke test successful. Profile keys: %s", list(ctx["user_profile"].keys()))
    except ContextNotFoundError as e:
        logger.info(f"Expected miss (smoke test): {e}")
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")