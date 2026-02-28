import unittest

import pandas as pd

from src.data.wrangling import (
    add_outcome_flags,
    add_refi_pathway,
    apply_core_wrangling,
    leakage_exclusions,
    resolved_modeling_frame,
    summarize_outcomes,
    to_numeric_percent,
)


class WranglingFeatureTests(unittest.TestCase):
    def test_percent_parser(self):
        s = pd.Series(["13.5%", " 42 ", None, "abc"])
        parsed = to_numeric_percent(s)
        self.assertAlmostEqual(parsed.iloc[0], 13.5)
        self.assertAlmostEqual(parsed.iloc[1], 42.0)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))

    def test_outcome_flags(self):
        df = pd.DataFrame(
            {
                "loan_status": [
                    "Charged Off",
                    "Default",
                    "Current",
                    "In Grace Period",
                    "Fully Paid",
                ]
            }
        )
        out = add_outcome_flags(df)
        self.assertEqual(out["is_default"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(out["is_unresolved"].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(out["is_resolved"].tolist(), [1, 1, 0, 0, 1])

    def test_refi_pathway(self):
        df = pd.DataFrame({"purpose": ["credit_card", "debt_consolidation", "personal", "vacation"]})
        out = add_refi_pathway(df)
        self.assertEqual(out["refi_pathway"].astype(str).tolist(), ["cc_to_pl", "pl_refinance", "pl_refinance", "other"])

    def test_resolved_frame(self):
        df = pd.DataFrame(
            {
                "loan_status": ["Current", "Charged Off", "Fully Paid"],
                "purpose": ["credit_card", "debt_consolidation", "vacation"],
                "int_rate": ["10.0%", "20.0%", "12.0%"],
                "revol_util": ["50%", "70%", "10%"],
                "issue_d": ["Jan-2018", "Jan-2017", "Jan-2016"],
            }
        )
        wrangled = apply_core_wrangling(df)
        model_df = resolved_modeling_frame(wrangled)
        self.assertEqual(len(model_df), 2)

    def test_outcome_summary(self):
        df = pd.DataFrame({"loan_status": ["Charged Off", "Current", "Fully Paid", "Default"]})
        out = add_outcome_flags(df)
        summary = summarize_outcomes(out)
        self.assertEqual(summary.rows_total, 4)
        self.assertEqual(summary.rows_unresolved, 1)
        self.assertAlmostEqual(summary.default_rate_all, 0.5)

    def test_leakage_exclusions_contains_total_pymnt(self):
        excl = leakage_exclusions()
        self.assertTrue("total_pymnt" in set(excl["column"].tolist()))


if __name__ == "__main__":
    unittest.main()
