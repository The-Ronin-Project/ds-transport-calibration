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
        ) = transport_calibration_xgboost_wrapper_testing_utils.run_xgb_wrapper_with_shap(
            test_automatic_fit=True
        )

        # Check shap consistency
        for k in metrics.keys():
            if "sanity" in k:
                self.assertTrue(metrics[k])

    def test_e2e_xgb_one_cov_shap(self):
        # Loop over domain parameters
        for domain_params in ["c1", "c2", "c3"]:
            # Run end-to-end
            (
                metrics,
                model,
            ) = transport_calibration_xgboost_wrapper_testing_utils.run_xgb_wrapper_with_shap_for_one_cov(
                predefined_domain_param_set=domain_params
            )

            # Check shap consistency
            for k in metrics.keys():
                if "sanity" in k:
                    self.assertTrue(metrics[k])

    def test_e2e_xgb_one_cov_shap_autofit(self):
        # Run end-to-end
        (
            metrics,
            model,
        ) = transport_calibration_xgboost_wrapper_testing_utils.run_xgb_wrapper_with_shap_for_one_cov(
            test_automatic_fit=True
        )

        # Check shap consistency
        for k in metrics.keys():
            if "sanity" in k:
                self.assertTrue(metrics[k])
