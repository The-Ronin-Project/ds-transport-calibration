import unittest
import numpy
from tests import transport_calibration_testing_utils


class TestTransportCalibration(unittest.TestCase):
    def test_output_shape_binary_histogram(self):
        # Simulate data and train a calibrator
        results, calibrator = transport_calibration_testing_utils.run_e2e(
            2, "logistic", ratio_estimator="histogram"
        )

        # Check that (N,2) shaped input produces correct output shape
        res1 = calibrator.calibrated_probability(
            numpy.asarray([0.65, 0.35] * 3).reshape(3, 2), 0.75
        )
        self.assertTrue(res1.shape == (3, 2))

        # Check that (N,) shaped input produces correct output shape
        res2 = calibrator.calibrated_probability(numpy.asarray([0.65, 0.5, 0.35]), 0.75)
        self.assertTrue(res2.shape == (3,))

        # Check that float input produces float output
        res3 = calibrator.calibrated_probability(0.35, 0.75)
        self.assertTrue(isinstance(res3, float))

        # Check that the outputs match at the expected array locations
        self.assertTrue(numpy.isclose(res1[0, 1], res2[2]))
        self.assertTrue(numpy.isclose(res1[0, 1], res3))

    def test_output_shape_binary_logistic(self):
        # Simulate data and train a calibrator
        results, calibrator = transport_calibration_testing_utils.run_e2e(
            2, "logistic", ratio_estimator="logistic"
        )

        # Check that (N,2) shaped input produces correct output shape
        res1 = calibrator.calibrated_probability(
            numpy.asarray([0.6, 0.4] * 3).reshape(3, 2), 0.7
        )
        self.assertTrue(res1.shape == (3, 2))

        # Check that (N,) shaped input produces correct output shape
        res2 = calibrator.calibrated_probability(numpy.asarray([0.6, 0.5, 0.4]), 0.7)
        self.assertTrue(res2.shape == (3,))

        # Check that float input produces float output
        res3 = calibrator.calibrated_probability(0.4, 0.7)
        self.assertTrue(isinstance(res3, float))

        # Check that the outputs match at the expected array locations
        self.assertTrue(numpy.isclose(res1[0, 1], res2[2]))
        self.assertTrue(numpy.isclose(res1[0, 1], res3))

    def test_output_shape_multiclass(self):
        # Simulate data and train a calibrator
        results, calibrator = transport_calibration_testing_utils.run_e2e(3, "logistic")

        # Check that (N,C) shaped input produces correct output shape
        res1 = calibrator.calibrated_probability(
            numpy.asarray([0.8, 0.1, 0.1, 0.3, 0.3, 0.4]).reshape(2, 3),
            numpy.asarray([0.4, 0.4, 0.2]),
        )
        self.assertTrue(res1.shape == (2, 3))

        # Check that (C,) shaped input produces correct output shape
        res2 = calibrator.calibrated_probability(
            numpy.asarray([0.3, 0.3, 0.4]), numpy.asarray([0.4, 0.4, 0.2])
        )
        self.assertTrue(res2.shape == (3,))

        # Check that the outputs match at the expected array locations
        self.assertTrue(numpy.isclose(res1[1, 0], res2[0]))
        self.assertTrue(numpy.isclose(res1[1, 1], res2[1]))
        self.assertTrue(numpy.isclose(res1[1, 2], res2[2]))

    def test_e2e_binary_histogram(self):
        # Run end-to-end and check basic parameters
        results, calibrator = transport_calibration_testing_utils.run_e2e(
            2, "logistic", ratio_estimator="histogram"
        )
        self.assertTrue(len(results["class_probability"]) == 2)
        self.assertTrue(calibrator.n_classes == 2)

        # Check that calibration succeeded
        self.assertTrue(
            numpy.isclose(
                results["class_probability"],
                results["predicted_prevalence_calibrated"],
                atol=0.1,
            ).all()
        )

    def test_e2e_multiclass_logistic(self):
        # Run end-to-end and check basic parameters
        results, calibrator = transport_calibration_testing_utils.run_e2e(7, "logistic")
        self.assertTrue(len(results["class_probability"]) == 7)
        self.assertTrue(calibrator.n_classes == 7)

        # Check that calibration succeeded
        self.assertTrue(
            numpy.isclose(
                results["class_probability"],
                results["predicted_prevalence_calibrated"],
                atol=0.1,
            ).all()
        )

    def test_e2e_multiclass_xgb(self):
        # Run end-to-end and check basic parameters
        results, calibrator = transport_calibration_testing_utils.run_e2e(4, "xgboost")
        self.assertTrue(len(results["class_probability"]) == 4)
        self.assertTrue(calibrator.n_classes == 4)

        # Check that calibration succeeded
        self.assertTrue(
            numpy.isclose(
                results["class_probability"],
                results["predicted_prevalence_calibrated"],
                atol=0.1,
            ).all()
        )
