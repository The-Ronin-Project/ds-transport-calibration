import unittest
from tests import transport_calibration_xgboost_wrapper_testing_utils


class TestTransportCalibrationXGBoostWrapper(unittest.TestCase):
    def test_e2e_xgb_shap(self):
        # Run end-to-end
        (
            metrics,
            model,
        ) = (
            transport_calibration_xgboost_wrapper_testing_utils.run_xgb_wrapper_with_shap()
        )

        # Check shap consistency
        for k in metrics.keys():
            if "sanity" in k:
                self.assertTrue(metrics[k])

    def test_e2e_xgb_shap_autofit(self):
        # Run end-to-end
        (
            metrics,
            model,
        ) = (
            transport_calibration_xgboost_wrapper_testing_utils.run_xgb_wrapper_with_shap(test_automatic_fit=True)
        )

        # Check shap consistency
        for k in metrics.keys():
            if "sanity" in k:
                self.assertTrue(metrics[k])
