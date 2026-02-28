import unittest

import pandas as pd

from src.models.search import classification_candidates, make_preprocessor, regression_candidates


class ModelUtilityTests(unittest.TestCase):
    def test_candidate_registry_contains_core_models(self):
        cls = classification_candidates(enable_svm=True)
        reg = regression_candidates(enable_svm=True)
        self.assertTrue("logistic_l2" in cls)
        self.assertTrue("random_forest" in cls)
        self.assertTrue("ridge" in reg)
        self.assertTrue("lasso" in reg)

    def test_preprocessor_builds_without_error(self):
        pre = make_preprocessor(
            numeric_cols=["annual_inc", "dti"],
            categorical_cols=["term", "grade"],
        )
        X = pd.DataFrame(
            {
                "annual_inc": [50000, 70000, None],
                "dti": [15.2, None, 22.1],
                "term": ["36 months", "60 months", "36 months"],
                "grade": ["B", "C", None],
            }
        )
        Xt = pre.fit_transform(X)
        self.assertEqual(Xt.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
