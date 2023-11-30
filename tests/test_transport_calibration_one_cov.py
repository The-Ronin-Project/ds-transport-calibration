import unittest
import numpy
from tests import transport_calibration_one_cov_testing_utils


class TestTransportCalibrationOneCov(unittest.TestCase):
    def test_output_shape_binary(self):
        # Simulate data and train a calibrator
        results, calibrator = transport_calibration_one_cov_testing_utils.run_e2e(n_classes=2)

        # Check that (N,2) shaped input produces correct output shape
        res1 = calibrator.calibrated_probability(
            numpy.asarray([0.6, 0.4] * 3).reshape(3, 2), numpy.asarray([0.7]*3)
        )
        self.assertTrue(res1.shape == (3, 2))

        # Check that (N,) shaped input produces correct output shape
        res2 = calibrator.calibrated_probability(numpy.asarray([0.6, 0.5, 0.4]), numpy.asarray([0.7]*3))
        self.assertTrue(res2.shape == (3,))

        # Check that float input produces float output
        res3 = calibrator.calibrated_probability(0.4, 0.7)
        self.assertTrue(isinstance(res3, float))

        # Check that the outputs match at the expected array locations
        self.assertTrue(numpy.isclose(res1[0, 1], res2[2]))
        self.assertTrue(numpy.isclose(res1[0, 1], res3))


    def test_e2e_binary(self):
        # Run end-to-end and check basic parameters
        results, calibrator = transport_calibration_one_cov_testing_utils.run_e2e(n_classes=2)
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