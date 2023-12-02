import warnings
import numpy
from . import transport_calibration, transport_calibration_one_cov
import xgboost
import xgboost.sklearn

# Try to handle different versions of xgboost
try:
    from xgboost.sklearn import _SklObjective
except ImportError:
    from xgboost.sklearn import SklObjective as _SklObjective


class TransportCalibration_XGBClassifier_Base(xgboost.XGBClassifier):
    def __init__(self, *, objective: _SklObjective = "binary:logistic", **kwargs):
        self._automatically_fit_calibrator_at_model_fit = kwargs.pop(
            "automatically_fit_calibrator_at_model_fit", False
        )
        super().__init__(objective=objective, **kwargs)
        self._transport_calibration_calibrator = None

    def transport_calibration_fit(self, *args, **kwargs):
        ...

    def transport_calibration_predict_proba_uncalibrated(self, *args, **kwargs):
        """Access to uncalibrated version of predict_proba"""
        return super().predict_proba(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """Fit the model and optionally automatically fit the calibrator"""
        # Fit the model using the fit method from the base class
        super().fit(*args, **kwargs)

        # If requested, automatically fit the calibrator and initialize it to be runnable
        if self._automatically_fit_calibrator_at_model_fit:
            # Fit the calibrator with the same data
            self.transport_calibration_fit(args[0], args[1], make_runable=True)
        return self

    @property
    def estimator(self):
        """Make this class compatible as a drop-in replacement to CalibratedClassiferCV from sklearn"""
        return self

    def count_classes_and_check_transport_calibration_inputs(
        self,
        training_labels,
    ):
        """Check the inputs to the transport calibration fitter to ensure it is ready to fit

        training_labels -- numpy array of shape (N,) containing an integer class label from 0 to C-1 for each of the C classes

        Returns the number of classes that will be fit

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
        return n_classes

    def transport_calibration_raise_if_not_fit(self):
        """Check that the calibrator is fit"""
        if self._transport_calibration_calibrator is None:
            raise ValueError(
                "Need to first fit the calibrator by calling transport_calibration_fit"
            )


class TransportCalibration_XGBClassifier(TransportCalibration_XGBClassifier_Base):
    def __init__(self, *, objective: _SklObjective = "binary:logistic", **kwargs):
        super().__init__(objective=objective, **kwargs)
        self._transport_calibration_class_probability = None

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
        make_runable=False,
    ):
        """Fit the calibrator using the provided training data

        training_features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features
        training_labels -- numpy array of shape (N,) containing an integer class label from 0 to C-1 for each of the C classes
        training_class_probability -- a numpy array of shape (C,) containing the class prevalence
                                      if None then compute it from the training_labels
        ratio_estimator -- string indicating which density estimator to use:
                           'histogram' only works for binary classification
                           'logistic' for any dimensionality
        make_runable -- if True, init target class-probability from the training data so that calibrator can run immediately

        if ratio_estimator is None, then automatically use 'histogram' for binary classification and 'logistic' for multi-class

        """
        # Check state and determine the number of classes from the parent class
        n_classes = self.count_classes_and_check_transport_calibration_inputs(
            training_labels
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

        # Initialize the target class prevalence with the training values if desired
        if make_runable:
            self.transport_calibration_class_probability = None

    def predict_proba(self, features):
        """Predict the calibrated probability

        features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features

        """
        # Check that the object is initialized for calibrated outputs
        self.transport_calibration_raise_if_not_fit()
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


class TransportCalibrationOneCov_XGBClassifier(TransportCalibration_XGBClassifier_Base):
    def __init__(self, *, objective: _SklObjective = "binary:logistic", **kwargs):
        self._transport_calibration_one_cov_feature_index = kwargs.pop(
            "transport_calibration_one_cov_feature_index", -1
        )
        super().__init__(objective=objective, **kwargs)

    @property
    def ocf_index(self):
        """The index of the one_cov_feature in the full feature-array"""
        return self._transport_calibration_one_cov_feature_index

    def extract_one_cov_from_full_features(self, features):
        """Extract the adjustment covariate from the full feature-array"""
        try:
            return features[:, self.ocf_index].flatten()
        except IndexError:
            raise ValueError(
                f"Invalid index={self.ocf_index} for adjustment covariate in feature array."
            )

    def transport_calibration_fit(
        self,
        training_features,
        training_labels,
        labels_primed=None,
        xvals_primed=None,
        make_runable=False,
    ):
        """Fit the calibrator using the provided training data

        training_features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features
        training_labels -- numpy array of shape (N,) containing an integer class label from 0 to C-1 for each of the C classes
        labels_primed -- numpy array of length M containing labels in the primed domain
        xvals_primed -- numpy array of length M containing a float value for X at each example in the primed domain
        make_runable -- if True, init target dist-data from the training data so that calibrator can run immediately

        Note: if valid data is provided for 'labels_primed' and 'xvals_primed', then the calibrator
        will be made runable using that data regardless of the value of 'make_runable'

        """
        # Check state and determine the number of classes from the parent class
        self.count_classes_and_check_transport_calibration_inputs(training_labels)

        # Extract the adjustment covariate
        xvals = self.extract_one_cov_from_full_features(training_features)

        # Compute scores to be used for training
        training_scores = self.transport_calibration_predict_proba_uncalibrated(
            training_features
        )

        # Instantiate and train the calibrator
        self._transport_calibration_calibrator = (
            transport_calibration_one_cov.TransportCalibrationOneCov(
                training_scores,
                training_labels,
                xvals,
            )
        )

        # Initialize the target data-distribution with the training values if desired
        if (
            not isinstance(labels_primed, numpy.ndarray)
            and not isinstance(xvals_primed, numpy.ndarray)
            and make_runable
        ):
            self._transport_calibration_calibrator.set_primed_distribution(
                training_labels, xvals
            )

        # Initialize the target data-distribution using the provided data if present
        if isinstance(labels_primed, numpy.ndarray) and isinstance(
            xvals_primed, numpy.ndarray
        ):
            self._transport_calibration_calibrator.set_primed_distribution(
                labels_primed, xvals_primed
            )

    def set_primed_distribution(self, labels_primed, xvals_primed):
        """Set the primed distribution using the provided data from the target domain

        labels_primed -- numpy array of length M containing labels in the primed domain
        xvals_primed -- numpy array of length M containing a float value for X at each example in the primed domain

        This function may be used as an alternate to provided this data when initially fitting the calibrator, or
        to update the distribution according to new target-domain data

        """
        self._transport_calibration_calibrator.set_primed_distribution(
            labels_primed, xvals_primed
        )

    def predict_proba(self, features):
        """Predict the calibrated probability

        features -- numpy array of shape (N,F) where N is the number of rows and F is the number of features

        """
        # Check that the object is initialized for calibrated outputs
        self.transport_calibration_raise_if_not_fit()
        if not self._transport_calibration_calibrator.ready_for_inference:
            raise ValueError("Calibrator not ready for inference.")

        # Extract the adjustment covariate
        xvals = self.extract_one_cov_from_full_features(features)

        # Compute uncalibrated scores
        scores = self.transport_calibration_predict_proba_uncalibrated(features)

        # Calibrate and return the adjusted probabilities
        return self._transport_calibration_calibrator.calibrated_probability(
            scores, xvals
        )
