import numpy
import sklearn
import sklearn.linear_model


class TransportCalibrationOneCov:
    def __init__(self, raw_pred, labels, xvals, labels_primed=None, xvals_primed=None):
        """Initialize the calibration adjustment

        raw_pred -- numpy array containing the model-score vectors with shape (N,C) (N-number of examples, C-number of classes)
        labels -- numpy array of length N containing an integer 0 to C-1 indicating the label for each row of 'raw_pred'
        xvals -- numpy array of length N containing a float value for X at each example
        labels_primed -- (optional) numpy array of length M containing labels in the primed domain
        xvals_primed -- (optional) numpy array of length M containing a float value for X at each example in the primed domain

        Note: the labels array must include some examples of every class

        Note: it is optional to provide the primed-domain data during construction, but if it is None,
              then you must provide it later by calling set_primed_distribution(...) before running inference

        Note: for binary classification, this class accepts simplified array shapes as follows:
              raw_pred may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)

        """
        # Check inputs and reshape as necessary
        if not (
            isinstance(raw_pred, numpy.ndarray)
            and isinstance(labels, numpy.ndarray)
            and isinstance(xvals, numpy.ndarray)
        ):
            raise ValueError("Input values must be numpy arrays.")
        if len(raw_pred.shape) == 1 or raw_pred.shape[1] == 1:
            # For binary classification this is allowed: adjust the value and shape to be consistent with a multi-class input
            raw_pred = numpy.concatenate(
                [1 - raw_pred.reshape(-1, 1), raw_pred.reshape(-1, 1)], axis=1
            )

        # Store the number of classes
        unique_labels = numpy.unique(labels)
        max_label = max(unique_labels)
        len_unique = len(unique_labels)
        if len_unique != max_label + 1:
            raise ValueError(
                f"Seems like some label examples are missing: max label = {max_label}, length = {len_unique}"
            )
        self._n_classes = len_unique

        # Construct logistic regression models needed for computing density ratios
        self._PY_RX = sklearn.linear_model.LogisticRegression().fit(
            numpy.concatenate([raw_pred, xvals.reshape(-1,1)], axis=1), labels
        )
        self._PY_X = sklearn.linear_model.LogisticRegression().fit(xvals.reshape(-1,1), labels)

        # Process and store the primed distribution if data was provided or mark it as missing for now
        if isinstance(labels_primed, numpy.ndarray) and isinstance(
            xvals_primed, numpy.ndarray
        ):
            self.set_primed_distribution(labels_primed, xvals_primed)
        else:
            self._PY_X_primed = None

    @property
    def n_classes(self):
        """Number of classes that this calibrator was initialized for (immutable)"""
        return self._n_classes

    @property
    def ready_for_inference(self):
        """Check that the object is fully initialized and ready for inference"""
        if self._PY_X_primed is not None:
            return True
        else:
            return False

    def set_primed_distribution(self, labels_primed, xvals_primed):
        """Fit the primed data and store the distribution for use during inference

        labels_primed -- numpy array of length M containing labels in the primed domain
        xvals_primed -- numpy array of length M containing a float value for X at each example in the primed domain

        """
        # Check inputs
        if not (
            isinstance(labels_primed, numpy.ndarray)
            and isinstance(xvals_primed, numpy.ndarray)
        ):
            raise ValueError("Input values must be numpy arrays.")
        unique_labels_primed = numpy.unique(labels_primed)
        max_label_primed = max(unique_labels_primed)
        len_unique_primed = len(unique_labels_primed)
        if len_unique_primed != max_label_primed + 1:
            raise ValueError(
                f"Seems like some label_primed examples are missing: max label = {max_label_primed}, length = {len_unique_primed}"
            )
        if len_unique_primed != self.n_classes:
            raise ValueError(
                f"Inconsistent number of classes in primed domain: {self.n_classes} vs. {len_unique_primed} in primed domain"
            )

        # Learn the distribution
        self._PY_X_primed = sklearn.linear_model.LogisticRegression().fit(
            xvals_primed.reshape(-1,1), labels_primed
        )

    def calibrated_probability(self, scores, xvals):
        """Compute P'(Y=c | R,X) ie. the posterior probability that Y is class c

        scores -- numpy array containing the raw model-scores with shape (N,C) (N-number of examples, C-number of classes)
        xvals -- numpy array containing the value of X for each example with shape (N,)

        Note: for binary classification, this class accepts simplified array shapes a follows:
            scores may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)


        Shape of the output depends on the shape of the input 'scores'

            The shape of 'xvals' does not affect the output shape. But 'xvals' shape must be congruent with 'scores' shape.
            Unless otherwise noted, 'xvals' must have shape (N,)

            Multi-class (n_classes >= 3):
                if scores.shape is (N,C) then output will have shape (N,C)      (corresponds to N examples)
                if scores.shape is (C,) then then output will have shape (C,)   (corresponds to a single example)
                    and xvals may be a float or have shape (1,)

            Binary (n_classes == 2):
                if scores.shape is (N,2) then output will have shape (N,2)      (corresponds to multi-class case)

                if scores.shape is (N,) then output will have shape (N,), and each value will be the C=1 value
                if scores is float then the output will be float, and will be the C=1 value
                    and xvals must also be a float
        """
        # Ensure that this object is fully initialized for inference
        if not self.ready_for_inference:
            raise ValueError("Not initialized for inference.")

        # Define default output flags which control the shape of the output according to the input (see the docstring)
        scalar_output = False
        flatten_output = False
        slice_class_1_output = False

        # Check inputs and reshape as necessary
        if self.n_classes == 2:
            # Identify invalid inputs
            if isinstance(scores, numpy.ndarray):
                if (len(scores.shape) == 2 and scores.shape[1] > 2) or len(
                    scores.shape
                ) > 2:
                    raise ValueError(
                        "Trained for binary classification but got unexpected shape = {shape} for input scores.".format(
                            shape=scores.shape
                        )
                    )

            # Identify variations of allowed inputs
            if isinstance(scores, float):
                # Check that other inputs are compatible
                if not isinstance(xvals, float):
                    raise ValueError(
                        "Inconsistent types for 'scores' and 'xvals' : if 'scores' is float then 'xvals' must also be."
                    )

                # This is the 'scores is float' case mentioned in the docstring
                scalar_output = True
                shaped_scores = numpy.asarray([1 - scores, scores]).reshape(1, 2)
                shaped_xvals = numpy.asarray([xvals])
        if not scalar_output and not isinstance(scores, numpy.ndarray):
            raise ValueError("Input values must be numpy arrays.")

        # Determine the shape of the output according to the shape of the input and the type of model
        if self.n_classes == 2 and not scalar_output and len(scores.shape) == 1:
            # This is the scores.shape is (N,) case mentioned in the docstring
            shaped_scores = numpy.concatenate(
                [1 - scores.reshape(-1, 1), scores.reshape(-1, 1)], axis=1
            )
            slice_class_1_output = True

            # Confirm that xvals is an appropriate shape
            if (
                isinstance(xvals, numpy.ndarray)
                and len(xvals.shape) == 1
                and xvals.shape[0] == scores.shape[0]
            ):
                shaped_xvals = xvals
            else:
                raise ValueError(
                    "Inconsistent types/shapes between 'scores' and 'xvals'"
                )
        elif not scalar_output:
            # Handle the shapes of either a single score input or multiple
            if len(scores.shape) == 1:
                # This is the scores.shape is (C,) case mentioned in the docstring
                shaped_scores = scores.reshape(-1, self.n_classes)
                flatten_output = True

                # Adjust xvals as necessary for the accepted input shapes
                if isinstance(xvals, float):
                    shaped_xvals = numpy.asarray([xvals])
                elif (
                    isinstance(xvals, numpy.ndarray)
                    and len(xvals.shape) == 1
                    and xvals.shape[0] == 1
                ):
                    shaped_xvals = xvals
                else:
                    raise ValueError(
                        "Inconsistent types/shapes between 'scores' and 'xvals'"
                    )
            else:
                # This is the scores.shape is (N,C) case mentioned in the docstring
                shaped_scores = scores
                shaped_xvals = xvals
        shaped_xvals = shaped_xvals.reshape(-1,1)

        # Compute the density ratios: output has N rows to cover each example in 'scores' and C columns for the classes
        py_x = self._PY_X.predict_proba(shaped_xvals)
        density_ratios = (
            self._PY_RX.predict_proba(
                numpy.concatenate([shaped_scores, shaped_xvals], axis=1)
            )
            / py_x
        )

        # Compute the conditional class-probability ratio differences (an NxCxC matrix)
        py_x_primed = self._PY_X_primed.predict_proba(shaped_xvals)
        w = numpy.zeros((density_ratios.shape[0], self.n_classes, self.n_classes))
        for c in range(self.n_classes):
            for chi in range(self.n_classes):
                if c != chi:
                    w[:, c, chi] = (py_x_primed[:, chi] / py_x_primed[:, c]) - (
                        py_x[:, chi] / py_x[:, c]
                    )

        # Evaluate the sum in the denominator for each class
        denom_sum = []
        for c in range(self.n_classes):
            denom_sum.append(numpy.sum(w[:, c, :] * density_ratios, axis=1))

        # Output has N rows to cover each example in 'scores' and C columns for each of the class
        denom_sum = numpy.asarray(denom_sum).transpose()

        # Assemble the rest of the denominator
        denom = 1 + py_x * denom_sum

        # Compute calibrated probability: output has N rows to cover each example in 'scores' and C columns for the classes
        posterior_probability = density_ratios * py_x / denom

        # Return result with matching shape to input
        if scalar_output:
            return float(posterior_probability[:, 1].flatten()[0])
        elif slice_class_1_output:
            return posterior_probability[:, 1].flatten()
        elif flatten_output:
            return posterior_probability.flatten()
        else:
            return posterior_probability
