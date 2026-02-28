import unittest

import numpy as np
import pandas as pd

from src.recommender.policy import apply_recommendation_policy, policy_simulation_table


class RecommenderPolicyTests(unittest.TestCase):
    def test_policy_outputs_expected_columns(self):
        df = pd.DataFrame(
            {
                "term": ["36 months", "60 months", "36 months"],
                "installment": [300.0, 500.0, 200.0],
                "refi_pathway": ["cc_to_pl", "pl_refinance", "other"],
                "int_rate_num": [18.0, 12.0, 15.0],
            }
        )
        pd_score = np.array([0.10, 0.30, 0.15])
        pred_rate = np.array([10.0, 11.0, 12.0])

        out = apply_recommendation_policy(df, pd_score=pd_score, pred_rate=pred_rate)
        for col in [
            "pd_score",
            "pred_rate",
            "scenario_current_rate",
            "monthly_savings",
            "risk_adjusted_savings",
            "recommendation",
        ]:
            self.assertTrue(col in out.columns)
        self.assertEqual(len(out), 3)
        self.assertTrue(set(out["recommendation"]).issubset({"hold", "consider", "strong_recommend"}))

    def test_policy_simulation_table_non_empty(self):
        df = pd.DataFrame(
            {
                "pd_score": [0.1, 0.2, 0.35, 0.05],
                "monthly_savings": [200, 50, 500, -10],
            }
        )
        table = policy_simulation_table(
            df,
            hold_thresholds=[0.2, 0.25],
            min_savings_values=[0, 100],
        )
        self.assertFalse(table.empty)
        self.assertTrue("consider_rate" in table.columns)


if __name__ == "__main__":
    unittest.main()
