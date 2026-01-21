from dataclasses import dataclass
import numpy as np

@dataclass
class UserProfile:
    user_id: int
    segment: int  # 0 low, 1 medium, 2 high


class UserSimulator:
    """
    Stochastic user behavior model.
    - purchase probability depends on engagement + inactivity + incentive
    - churn probability depends on inactivity + fatigue + competitor pressure
    """

    def __init__(self, profile: UserProfile, rng: np.random.Generator):
        self.profile = profile
        self.rng = rng

        # Base values by segment
        self.base_purchase = {0: 0.06, 1: 0.14, 2: 0.22}[profile.segment]
        self.base_churn = {0: 0.20, 1: 0.11, 2: 0.06}[profile.segment]
        self.base_ltv = {0: 120.0, 1: 600.0, 2: 2200.0}[profile.segment]
        self.base_order = {0: 35.0, 1: 85.0, 2: 160.0}[profile.segment]

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def respond(
        self,
        days_inactive: int,
        engagement: float,
        cart_abandon: float,
        promo_fatigue: float,
        action: int,
        competitor_offer: float = 0.0,  # NEW (safe default)
    ):
        """
        Inputs are already in [0,1] except days_inactive which is [0, max].
        competitor_offer ∈ {0.0, 0.10, 0.15}
        Returns: will_purchase, will_churn, order_value, ltv
        """

        # ----------------------------
        # Incentive effects (OUR action)
        # ----------------------------
        # action: 0 none, 1 10%, 2 15%, 3 ship, 4 points, 5 bundle
        purchase_boost = [0.00, 0.10, 0.18, 0.07, 0.05, 0.14][action]
        churn_reduction = [0.00, 0.06, 0.09, 0.05, 0.03, 0.07][action]

        # ----------------------------
        # Inactivity & fatigue
        # ----------------------------
        inact = np.clip(days_inactive / 30.0, 0.0, 1.0)
        fatigue = np.clip(promo_fatigue, 0.0, 1.0)

        # ----------------------------
        # Competitor pressure
        # ----------------------------
        # Normalize competitor offer to [0,1]
        comp = np.clip(competitor_offer / 0.15, 0.0, 1.0)

        # ----------------------------
        # Purchase probability
        # ----------------------------
        purchase_logit = (
            -1.4
            + 2.0 * engagement
            - 2.2 * inact
            + 0.8 * cart_abandon
            + 1.6 * purchase_boost * (1.0 - 0.6 * fatigue)
            - 1.3 * comp                 # competitor steals attention
        )

        p_purchase = self._sigmoid(purchase_logit)
        p_purchase = np.clip(
            p_purchase + (self.base_purchase - 0.12),
            0.01,
            0.95,
        )

        # ----------------------------
        # Churn probability
        # ----------------------------
        churn_logit = (
            -1.0
            + 3.0 * inact
            + 1.2 * fatigue
            - 1.6 * churn_reduction
            - 1.0 * engagement
            + 1.4 * comp                 # competitor increases churn
        )

        p_churn = self._sigmoid(churn_logit)
        p_churn = np.clip(
            p_churn + (self.base_churn - 0.12),
            0.01,
            0.98,
        )

        # ----------------------------
        # Sample outcomes
        # ----------------------------
        will_purchase = (self.rng.random() < p_purchase)
        will_churn = (self.rng.random() < p_churn)

        # Order value with noise
        order_value = float(
            self.base_order * (1.0 + self.rng.normal(0.0, 0.15))
        )
        order_value = max(order_value, 5.0)

        return will_purchase, will_churn, order_value, float(self.base_ltv)
