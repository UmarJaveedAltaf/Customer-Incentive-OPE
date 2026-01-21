# src/environment/competitor.py

class CompetitorPolicy:
    """
    Simple rule-based competitor incentive strategy
    """

    def select_offer(self, customer_segment, days_inactive):
        """
        Returns competitor offer strength:
        0.0   -> no offer
        0.10  -> 10% discount
        0.15  -> 15% discount
        """
        if customer_segment == "high" and days_inactive > 3:
            return 0.15
        if days_inactive > 5:
            return 0.10
        return 0.0
