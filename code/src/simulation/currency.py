import sqlite3
import bisect
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class CurrencyConverter:
    """
    Pure-Python exchange rate converter.
    Loads rates from the SQLite database (populated from exchange_rates.csv) 
    and provides fast, deterministic currency conversions with date fallback.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Structure: {(from_curr, to_curr): [(date_str, rate), ...]} sorted by date
        self.rates: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        self.dates_index: Dict[Tuple[str, str], List[str]] = {}
        self._load_rates()

    def _load_rates(self):
        """Loads all exchange rates from the database into memory for O(log N) lookups."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Cannot load exchange rates.")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT rate_date, from_currency, to_currency, rate FROM exchange_rates WHERE rate IS NOT NULL")
            rows = cursor.fetchall()
            
            temp_rates: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
            for date_str, from_curr, to_curr, rate in rows:
                key = (from_curr, to_curr)
                if key not in temp_rates:
                    temp_rates[key] = []
                temp_rates[key].append((date_str, float(rate)))
            
            # Sort by date string (YYYY-MM-DD format sorts chronologically as strings)
            for key, value in temp_rates.items():
                value.sort(key=lambda x: x[0])
                self.rates[key] = value
                self.dates_index[key] = [item[0] for item in value]
                
            logger.info(f"Loaded {len(rows)} exchange rate records into memory.")
        finally:
            conn.close()

    def _lookup_rate(self, from_curr: str, to_curr: str, date: str) -> Optional[float]:
        """
        Finds the exchange rate for a given pair on or before the specified date.
        Uses binary search for efficient date fallback (e.g., weekends/holidays).
        """
        key = (from_curr, to_curr)
        if key not in self.rates:
            return None
        
        dates_list = self.dates_index[key]
        rates_list = self.rates[key]
        
        # bisect_right finds the insertion point. Subtracting 1 gives the closest prior date.
        idx = bisect.bisect_right(dates_list, date) - 1
        
        if idx >= 0:
            return rates_list[idx][1]
        return None

    def get_rate(self, from_currency: str, to_currency: str, date: str) -> float:
        """
        Gets the exchange rate from one currency to another on a specific date.
        Falls back to the closest previous date if the exact date is missing.
        Automatically calculates the inverse rate if the direct pair is missing.
        """
        if from_currency == to_currency:
            return 1.0
        
        # 1. Try direct lookup
        rate = self._lookup_rate(from_currency, to_currency, date)
        if rate is not None:
            return rate
            
        # 2. Try inverse lookup (e.g., if we need USD->ZAR but only have ZAR->USD)
        inverse_rate = self._lookup_rate(to_currency, from_currency, date)
        if inverse_rate is not None and inverse_rate != 0:
            return 1.0 / inverse_rate
            
        raise ValueError(
            f"No exchange rate found for {from_currency} to {to_currency} "
            f"on or before {date}. Please check exchange_rates.csv."
        )

    def convert(self, amount: float, from_currency: str, to_currency: str, date: str) -> float:
        """
        Converts an amount from one currency to another using the rate on the given date.
        """
        if amount == 0.0:
            return 0.0
            
        rate = self.get_rate(from_currency, to_currency, date)
        return amount * rate

    