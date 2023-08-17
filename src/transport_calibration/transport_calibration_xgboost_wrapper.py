import warnings
import numpy
from . import transport_calibration
import xgboost
import xgboost.sklearn

# Try to handle different versions of xgboost
try:
    from xgboost.sklearn import _SklObjective
except ImportError:
    from xgboost.sklearn import SklObjective as _SklObjective


class TransportCalibration_XGBClassifier(xgboost.XGBClassifier):
    def __init__(self, *, objective: _SklObjective = "binary:logistic", **kwargs):
        super().__init__(objective=objective, **kwargs)
        self._transport_calibration_calibrator = None
        self._transport_calibration_class_probability = None

    def transport_calibration_predict_proba_uncalibrated(self, *args, **kwargs):
        """Access to uncalibrated version of predict_proba"""
        return super().predict_proba(*args, **kwargs)

    @property
    def transport_calibration_class_probability(self):
        """The prevalence of each class in the target domain"""
        return self._transport_calibration_class_probability

    @transport_calibration_class_probability.setter
    def transport_calibration_class_probability(self, class_probability):
        """Set the value of class_probability to be used for calibrated predictions

        class_probability -- a numpy array of shape (C,) containing the class prevalence in the target domain

        Passing a value of None will set the class probability to the value from the training data

        """
        # Silently fail if the calibrator is not yet initialized
        if self._transport_calibration_calibrator is None:
            self._transport_calibration_class_probability = None
        else:
            # Use the training-domain class probability if desired
            if class_probability is None:
                class_probability = (
                    self._transport_calibration_calibrator.training_class_probability
                )

            # Store the set value, converting to a numpy array and renormalizing it if necessary
            self._transport_calibration_class_probability = (
                self._transport_calibration_calibrator._repair_class_probability(
                    numpy.asarray(class_probability).flatten()
                )
            )

    def transport_calibration_fit(
        self,
        training_features,
        training_labels,
        training_class_probability=None,
        ratio_estimator=None,
    ):
        """Fit the calibrator using the provided training data

        training_features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features
        training_labels -- numpy array of shape (N,) containing an integer class label from 0 to C-1 for each of the C classes
        training_class_probability -- a numpy array of shape (C,) containing the class prevalence
                                      if None then compute it from the training_labels
        ratio_estimator -- string indicating which density estimator to use:
                           'histogram' only works for binary classification
                           'logistic' for any dimensionality

        if ratio_estimator is None, then automatically use 'histogram' for binary classification and 'logistic' for multi-class

        """
        # Check state
        if self._transport_calibration_calibrator is not None:
            warnings.warn(
                "Calibrator is already trained- now retraining and overwriting"
            )

        # Determine the number of classes from the parent class
        try:
            n_classes = self.n_classes_
        except AttributeError:
            raise ValueError(
                "Does not seem that this classifier was trained yet- need to train it first."
            )

        # Check inputs
        if isinstance(training_labels, numpy.ndarray):
            # Check that labels fit within expected range
            if min(training_labels) < 0 or max(training_labels) >= n_classes:
                raise ValueError(
                    "Invalid range of training labels: must be from 0 to n_classes-1"
                )
        else:
            raise ValueError(
                "Invalid input type for training_labels: must be a numpy array"
            )

        # Compute class prevalence if desired
        if training_class_probability is None:
            # One-hot encode the labels and compute prevalence
            onehot = numpy.eye(n_classes)[training_labels]
            training_class_probability = onehot.mean(axis=0)

        # Select ratio estimator algorithm if desired
        if ratio_estimator is None:
            if n_classes == 2:
                ratio_estimator = "histogram"
            else:
                ratio_estimator = "logistic"

        # Compute scores to be used for training
        training_scores = self.transport_calibration_predict_proba_uncalibrated(
            training_features
        )

        # Instantiate and train the calibrator
        self._transport_calibration_calibrator = (
            transport_calibration.TransportCalibration(
                training_scores,
                training_labels,
                training_class_probability,
                ratio_estimator=ratio_estimator,
            )
        )

    def predict_proba(self, features):
        """Predict the calibrated probability

        features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features

        """
        # Check that the object is initialized for calibrated outputs
        if self._transport_calibration_calibrator is None:
            raise ValueError(
                "Need to first fit the calibrator by calling transport_calibration_fit"
            )
        if self.transport_calibration_class_probability is None:
            raise ValueError(
                "Need to set the value of transport_calibration_class_probability for the target domain"
            )

        # Compute uncalibrated scores
        scores = self.transport_calibration_predict_proba_uncalibrated(features)

        # Calibrate and return the adjusted probabilities
        return self._transport_calibration_calibrator.calibrated_probability(
            scores, self.transport_calibration_class_probability
        )