import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CashFlowSimulator:
    """
    Pure-Python cash flow simulator with a 90-day horizon (extendable for
    longer payment plans). Forecasts balances from the request date, converts
    foreign-currency events with the fixed dated rates, and computes the safe
    payment amount and the earliest safe full-payment date.
    """

    def __init__(self, currency_converter):
        self.converter = currency_converter

    # ──────────────────────────────────────────────
    # Duplicate detection
    # ──────────────────────────────────────────────
    @staticmethod
    def _chain_root(event_id: Optional[str], parent: Dict[str, str]) -> Optional[str]:
        seen = set()
        node = event_id
        while node and node in parent and node not in seen:
            seen.add(node)
            node = parent[node]
        return node

    def _resolve_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies the dataset rules: drops failed/cancelled/unrealized events and
        pending credits, then removes duplicate records from the same
        transaction lifecycle using the explicit `linked_event_id` links
        instead of guessing by (amount, category, direction).
        """
        candidates = []
        for ev in events:
            status = ev.get('status')
            direction = ev.get('direction')
            if status in ('failed', 'cancelled', 'unrealized'):
                continue
            if status == 'pending' and direction == 'credit':
                continue
            candidates.append(ev)

        id_to_ev = {ev.get('event_id'): ev for ev in candidates if ev.get('event_id')}
        parent = {}
        for ev in candidates:
            linked = ev.get('linked_event_id')
            if linked and linked in id_to_ev:
                parent[ev['event_id']] = linked

        chains: Dict[str, List[str]] = {}
        for ev in candidates:
            eid = ev.get('event_id')
            if not eid:
                continue
            root = self._chain_root(eid, parent) or eid
            chains.setdefault(root, []).append(eid)

        drop_ids = set()
        for root, members in chains.items():
            if len(members) < 2:
                continue
            by_direction: Dict[str, List[str]] = {}
            for eid in members:
                direction = id_to_ev[eid].get('direction')
                by_direction.setdefault(direction, []).append(eid)

            for _direction, eids in by_direction.items():
                group = [id_to_ev[eid] for eid in eids]
                settled = [e for e in group if e.get('status') == 'settled']
                pending = [e for e in group if e.get('status') == 'pending']

                # A pending copy of an already settled transaction is the same
                # money twice; keep the settled record.
                if settled and pending:
                    for e in pending:
                        drop_ids.add(e['event_id'])
                    continue

                # Two settled records of the same lifecycle that fall close in
                # time are duplicates too; keep the most recent one.
                if len(group) > 1:
                    dated = [
                        (e, e.get('settlement_date') or e.get('event_date') or "")
                        for e in group
                    ]
                    dated.sort(key=lambda pair: pair[1])
                    # Only collapse when the two records are within 45 days.
                    if dated[1][1] and dated[0][1]:
                        try:
                            close = (
                                datetime.strptime(dated[1][1], "%Y-%m-%d")
                                - datetime.strptime(dated[0][1], "%Y-%m-%d")
                            ).days <= 45
                        except ValueError:
                            close = False
                        if close:
                            for e, _ in dated[:-1]:
                                drop_ids.add(e['event_id'])

        resolved = [ev for ev in candidates if ev.get('event_id') not in drop_ids]
        return resolved

    # ──────────────────────────────────────────────
    # Core simulation
    # ──────────────────────────────────────────────
    def simulate(
        self,
        user_profile: Dict[str, Any],
        events: List[Dict[str, Any]],
        request_date_str: str,
        requested_amount: float,
        end_date: Optional[str] = None,
        spending_changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Simulates cash flow and computes safe-payment metrics.

        Args:
            user_profile: balance, minimum balance, home currency.
            events: financial events for the user.
            request_date_str: request date (YYYY-MM-DD).
            requested_amount: total amount the user wants to pay.
            end_date: optional horizon extension (YYYY-MM-DD) so long payment
                      plans are checked in full.
            spending_changes: optional spending changes which also cap the
                              category-level recurring forecasts.

        Returns:
            Dict with amount_safe_to_pay, earliest_date_for_full_payment,
            and min_future_balance.
        """
        request_date = datetime.strptime(request_date_str, "%Y-%m-%d").date()
        default_end = request_date + timedelta(days=90)
        if end_date:
            try:
                extension = datetime.strptime(end_date, "%Y-%m-%d").date()
                default_end = max(default_end, extension)
            except ValueError:
                pass

        home_currency = user_profile.get('home_currency', 'USD')
        current_balance = float(user_profile.get('current_available_balance', 0.0) or 0.0)
        min_balance = float(user_profile.get('minimum_balance_to_keep', 0.0) or 0.0)

        resolved_events = self._resolve_events(events)

        # Map cash flows to dates and convert currency.
        daily_flows: Dict[Any, float] = {}
        for ev in resolved_events:
            direction = ev.get('direction')
            if direction == 'debit':
                pass
            elif direction == 'credit' and ev.get('status') in ('settled', 'scheduled') and ev.get('category') == 'salary':
                pass
            else:
                continue

            amount = ev.get('amount')
            if amount is None:
                continue  # ignored blank amounts must be backfilled via OCR upstream

            currency = ev.get('currency') or home_currency
            event_date_str = ev.get('settlement_date') or ev.get('event_date')
            if not event_date_str:
                continue

            try:
                event_date = datetime.strptime(event_date_str, "%Y-%m-%d").date()
            except ValueError:
                continue

            if event_date < request_date or event_date > default_end:
                continue

            try:
                converted = self.converter.convert(float(amount), currency, home_currency, event_date_str)
            except ValueError as exc:
                logger.warning(f"Currency conversion failed for event {ev.get('event_id')}: {exc}")
                continue

            daily_flows[event_date] = daily_flows.get(event_date, 0.0) + (
                -converted if direction == 'debit' else converted
            )

        # Recurring-expense forecast: established debits/salary continue.
        from ..simulation.forecast import project_recurring
        projected = project_recurring(
            events, request_date_str, default_end.strftime("%Y-%m-%d"), spending_changes
        )
        for day, value in projected.items():
            daily_flows[day] = daily_flows.get(day, 0.0) + value

        total_days = (default_end - request_date).days + 1
        balances = []
        cum_flow = 0.0
        for i in range(total_days):
            day = request_date + timedelta(days=i)
            cum_flow += daily_flows.get(day, 0.0)
            balances.append((day, current_balance + cum_flow))

        # The amount safe on request_date is judged against the next pay cycle:
        # commitments until the next confirmed/recurring salary receipt.
        salary_cut_idx = total_days
        if not end_date and not spending_changes:
            salary_cut = self._next_salary_date(events, request_date)
            if salary_cut:
                cutdays = (salary_cut - request_date).days + 1
                if 0 < cutdays <= total_days:
                    salary_cut_idx = cutdays

        min_future_balance = min(balance for _, balance in balances)
        min_until_next_salary = min(balance for _, balance in balances[:salary_cut_idx])

        amount_safe_to_pay = min_until_next_salary - min_balance
        amount_safe_to_pay = max(0.0, amount_safe_to_pay)
        amount_safe_to_pay = min(amount_safe_to_pay, requested_amount)

        # Suffix minimums to find the first day a full payment stays safe.
        suffix_min = [0.0] * total_days
        suffix_min[total_days - 1] = balances[total_days - 1][1]
        for i in range(total_days - 2, -1, -1):
            suffix_min[i] = min(balances[i][1], suffix_min[i + 1])

        earliest_date = None
        if amount_safe_to_pay >= requested_amount:
            earliest_date = request_date_str
        else:
            for i in range(total_days):
                if suffix_min[i] - requested_amount >= min_balance:
                    earliest_date = balances[i][0].strftime("%Y-%m-%d")
                    break

        return {
            "amount_safe_to_pay": round(amount_safe_to_pay, 2),
            "earliest_date_for_full_payment": earliest_date,
            "min_future_balance": round(min_future_balance, 2),
            "min_next_salary_balance": round(
                min_until_next_salary if salary_cut_idx < total_days else min_future_balance, 2
            ),
        }

    @staticmethod
    def _next_salary_date(events: List[Dict[str, Any]], request_date) -> Optional[Any]:
        """Next projected salary receipt strictly after request_date."""
        try:
            from ..simulation.forecast import build_recurrence_patterns
        except ImportError:
            return None
        request_date_str = request_date.strftime("%Y-%m-%d")
        for pat in build_recurrence_patterns(events, request_date_str):
            if pat.get("category") == "salary" and pat.get("direction") == "credit":
                last = pat.get("last_date")
                interval = pat.get("interval_days")
                if last is None or not interval:
                    return None
                nxt = last
                while nxt <= request_date:
                    nxt = nxt + timedelta(days=interval)
                if nxt > request_date:
                    return nxt
        return None

    def is_plan_viable(
        self,
        user_profile: Dict[str, Any],
        events: List[Dict[str, Any]],
        request_date_str: str,
        requested_amount: float,
        schedule: List[Dict[str, Any]],
        spending_changes: List[Dict[str, Any]],
    ) -> bool:
        """
        Checks whether a payment schedule (plus optional spending changes)
        keeps the balance at or above the minimum for the entire horizon.
        """
        changed_ids = {c['event_id']: c for c in spending_changes}
        category_by_id = {ev.get('event_id'): ev.get('category') for ev in events}
        for change in spending_changes:
            change.setdefault('_category', category_by_id.get(change.get('event_id')))
        modified_events = []
        for ev in events:
            if ev.get('event_id') in changed_ids:
                change = changed_ids[ev['event_id']]
                if change.get('action') == 'stop':
                    continue
                if change.get('action') == 'reduce_to':
                    modified = dict(ev)
                    modified['amount'] = change.get('new_amount', 0.0)
                    modified_events.append(modified)
                    continue
            modified_events.append(ev)

        home_currency = user_profile.get('home_currency', 'USD')
        horizon = request_date_str
        for pay in schedule:
            modified_events.append({
                'event_id': f"proposed_payment_{pay['date']}_{pay['amount']}",
                'direction': 'debit',
                'amount': pay['amount'],
                'currency': home_currency,
                'settlement_date': pay['date'],
                'status': 'scheduled',
                'category': 'proposed_payment',
            })
            if pay['date'] > horizon:
                horizon = pay['date']

        min_required = float(user_profile.get('minimum_balance_to_keep', 0.0) or 0.0)
        result = self.simulate(
            user_profile, modified_events, request_date_str, requested_amount, end_date=horizon,
            spending_changes=spending_changes,
        )
        return result['min_future_balance'] >= min_required