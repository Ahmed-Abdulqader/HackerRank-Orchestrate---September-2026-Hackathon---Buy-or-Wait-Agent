import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PlanRanker:
    """
    Pure-Python 6-step plan ranking engine.
    Evaluates and ranks candidate payment plans based on strict business rules 
    and exact tie-breakers to ensure deterministic plan selection.
    """

    def rank_plans(
        self, 
        candidate_plans: List[Dict[str, Any]], 
        desired_completion_date: str
    ) -> Optional[Dict[str, Any]]:
        """
        Ranks a list of candidate plans and returns the single best plan.
        
        Args:
            candidate_plans: List of dicts, each representing a viable plan. 
                             Expected keys per plan:
                             - payment_option_id (str or int)
                             - method (str)
                             - payments (List[Dict]) containing 'date' and 'amount'
                             - spending_changes (List[Dict])
                             - total_amount_paid (float)
                             - completion_date (str, YYYY-MM-DD)
                             - start_date (str, YYYY-MM-DD)
            desired_completion_date: The target date to complete the payment (YYYY-MM-DD).
            
        Returns:
            The best ranked plan dictionary, or None if the candidate list is empty.
        """
        if not candidate_plans:
            logger.warning("No candidate plans provided to the ranker.")
            return None

        try:
            target_date = datetime.strptime(desired_completion_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid desired_completion_date format: {desired_completion_date}")
            target_date = datetime.max.date()

        def sort_key(plan: Dict[str, Any]):
            # 1. Completes by desired_completion_date (True -> 0, False -> 1)
            # Lower is better, so completing on time gets priority (0).
            comp_date_str = plan.get("completion_date")
            if comp_date_str:
                try:
                    comp_date = datetime.strptime(comp_date_str, "%Y-%m-%d").date()
                    completes_on_time = 0 if comp_date <= target_date else 1
                except ValueError:
                    completes_on_time = 1
            else:
                completes_on_time = 1

            # 2. Requires no spending changes (False -> 0, True -> 1)
            # Lower is better, so no changes gets priority (0).
            has_changes = 0 if not plan.get("spending_changes") else 1

            # 3. Minimizes total amount paid (ascending)
            total_paid = plan.get("total_amount_paid", float('inf'))

            # 4. Starts payment earlier (ascending)
            # Lower date is better.
            start_date_str = plan.get("start_date")
            if start_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
                except ValueError:
                    start_date = datetime.max.date()
            else:
                start_date = datetime.max.date()

            # 5. Uses fewer payments (ascending)
            # Lower count is better.
            num_payments = len(plan.get("payments", []))

            # 6. Lowest payment_option_id tie-breaker (ascending)
            # Extract the numeric suffix so option_10 sorts after option_2.
            option_id = plan.get("payment_option_id", "")
            match = re.search(r"(\d+)$", str(option_id))
            numeric_option_id = int(match.group(1)) if match else float('inf')

            return (
                completes_on_time,
                has_changes,
                total_paid,
                start_date,
                num_payments,
                numeric_option_id
            )

        # Sort the plans using the strict 6-step tie-breaker logic
        sorted_plans = sorted(candidate_plans, key=sort_key)
        
        best_plan = sorted_plans[0]
        logger.info(
            f"Ranked {len(candidate_plans)} candidate plans. "
            f"Selected best plan: {best_plan.get('payment_option_id')} "
            f"(Method: {best_plan.get('method')}, Total: {best_plan.get('total_amount_paid')})"
        )
        
        return best_plan