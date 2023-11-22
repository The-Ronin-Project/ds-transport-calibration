import numpy
import sklearn
import sklearn.linear_model


class TransportCalibrationOneCov:
    def __init__(self, raw_pred, labels, xvals, labels_prime, xvals_prime):
        """Initialize the calibration adjustment

        raw_pred -- numpy array containing the model-score vectors with shape (N,C) (N-number of examples, C-number of classes)
        labels -- numpy array of length N containing an integer 0 to C-1 indicating the label for each row of 'raw_pred'
        xvals -- numpy array of length N containing a float value for X at each example
        labels_prime -- numpy array of length M containing labels in the primed domain
        xvals_prime -- numpy array of length M containing a float value for X at each example in the primed domain

        Note: the labels array must include some examples of every class

        Note: for binary classification, this class accepts simplified array shapes as follows:
            raw_pred may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)

        """
        # Check inputs and reshape as necessary
        if not (
            isinstance(raw_pred, numpy.ndarray)
            and isinstance(labels, numpy.ndarray)
            and isinstance(xvals, numpy.ndarray)
            and isinstance(labels_prime, numpy.ndarray)
            and isinstance(xvals_prime, numpy.ndarray)
        ):
            raise ValueError("Input values must be numpy arrays.")
        if len(raw_pred.shape) == 1 or raw_pred.shape[1] == 1:
            # For binary classification this is allowed: adjust the value and shape to be consistent with a multi-class input
            raw_pred = numpy.concatenate(
                [1 - raw_pred.reshape(-1, 1), raw_pred.reshape(-1, 1)], axis=1
            )

        # Store the number of classes
        unique_labels = numpy.unique(labels)
        unique_labels_prime = numpy.unique(labels_prime)
        max_label = max(unique_labels)
        max_label_prime = max(unique_labels_prime)
        len_unique = len(unique_labels)
        len_unique_prime = len(unique_labels_prime)
        if len_unique != max_label + 1:
            raise ValueError(f"Seems like some label examples are missing: max label = {max_label}, length = {len_unique}")
        if len_unique_prime != max_label_prime + 1:
            raise ValueError(f"Seems like some label_prime examples are missing: max label = {max_label_prime}, length = {len_unique_prime}")
        if len_unique != len_unique_prime:
            raise ValueError(f"Inconsistent number of labels between domains: {len_unique} vs. {len_unique_prime} in primed domain")
        self._n_classes = len_unique

        # Construct logistic regression models needed for computing density ratios
        self._PY_RX = sklearn.linear_model.LogisticRegression().fit(numpy.concatenate([raw_pred, xvals], axis=1), labels)
        self._PY_X = sklearn.linear_model.LogisticRegression().fit(xvals, labels)
        self._PY_X_prime = sklearn.linear_model.LogisticRegression().fit(xvals_prime, labels_prime)


    @property
    def n_classes(self):
        """Number of classes that this calibrator was initialized for (immutable)"""
        return self._n_classes


    def calibrated_probability(self, scores, xvals):
        """Compute P'(Y=c | R,X) ie. the posterior probability that Y is class c

        scores -- numpy array containing the raw model-scores with shape (N,C) (N-number of examples, C-number of classes)
        xvals -- numpy array containing the value of X for each example with shape (N,)

        Note: for binary classification, this class accepts simplified array shapes a follows:
            scores may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)


        Shape of the output depends on the shape of the input 'scores'

            Multi-class (n_classes >= 3):
                if scores.shape is (N,C) then output will have shape (N,C)      (corresponds to N examples)
                if scores.shape is (C,) then then output will have shape (C,)   (corresponds to a single example)

            Binary (n_classes == 2):
                if scores.shape is (N,2) then output will have shape (N,2)      (corresponds to multi-class case)

                if scores.shape is (N,) then output will have shape (N,), and each value will be the C=1 value
                if scores is float then the output will be float, and will be the C=1 value
        """
        # Define default output flags which control the shape of the output according to the input (see the docstring)
        scalar_output = False
        flatten_output = False
        slice_class_1_output = False

        # Check inputs and reshape as necessary
        if self.n_classes == 2:
            # Identify invalid inputs
            if (
                isinstance(class_probability, numpy.ndarray)
                and len(class_probability) > 2
            ):
                raise ValueError(
                    "Trained for binary classification but got unexpected shape = {shape} for class probability.".format(
                        shape=class_probability.shape
                    )
                )
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
            if (
                isinstance(class_probability, numpy.ndarray)
                and len(class_probability.shape) == 1
                and class_probability.shape[0] == 1
            ):
                # Allowed for binary classification: temporarily replace value so that it will get adjusted in the next check
                class_probability = float(class_probability[0])
            if isinstance(class_probability, float):
                # Allowed for binary classification: adjust the value and shape to be consistent with a multi-class input
                class_probability = numpy.asarray(
                    [1 - class_probability, class_probability]
                )
            if isinstance(scores, float):
                # This is the 'scores is float' case mentioned in the docstring
                scalar_output = True
                shaped_scores = numpy.asarray([1 - scores, scores]).reshape(1, 2)
        if not scalar_output and not (
            isinstance(scores, numpy.ndarray)
            and isinstance(class_probability, numpy.ndarray)
        ):
            raise ValueError("Input values must be numpy arrays.")

        # Repair class_probability if it contains any exactly 0 or negative values
        class_probability = self._repair_class_probability(class_probability)

        # Determine the shape of the output according to the shape of the input and the type of model
        if self.n_classes == 2 and not scalar_output and len(scores.shape) == 1:
            # This is the scores.shape is (N,) case mentioned in the docstring
            shaped_scores = numpy.concatenate(
                [1 - scores.reshape(-1, 1), scores.reshape(-1, 1)], axis=1
            )
            slice_class_1_output = True
        elif not scalar_output:
            # Handle the shapes of either a single score input or multiple
            if len(scores.shape) == 1:
                # This is the scores.shape is (C,) case mentioned in the docstring
                shaped_scores = scores.reshape(-1, self.n_classes)
                flatten_output = True
            else:
                # This is the scores.shape is (N,C) case mentioned in the docstring
                shaped_scores = scores

        # Compute the calibrated probability using desired ratio estimator
        if self.ratio_estimator == "histogram":
            # Evaluate the posterior probability: this ratio estimator only supports binary classifiers
            prior_ratio = (1 - class_probability[1]) / (
                1 - self.training_class_probability[1]
            )
            posterior_probability = numpy.clip(
                class_probability[1]
                / (
                    class_probability[1]
                    + (
                        self._ratio(shaped_scores[:, 1].flatten())
                        - self.training_class_probability[1]
                    )
                    * prior_ratio
                ),
                numpy.finfo(float).resolution,
                1 - numpy.finfo(float).resolution,
            )

            # Reshape to (N,2) so that the output shape can be modified at end of this method via a general procedure
            posterior_probability = numpy.concatenate(
                [
                    1 - posterior_probability.reshape(-1, 1),
                    posterior_probability.reshape(-1, 1),
                ],
                axis=1,
            )
        elif self.ratio_estimator == "logistic":
            # Precompute the density ratios: output has N rows to cover each example in 'scores' and C columns for the classes
            density_ratios = self._ratios(shaped_scores)

            # Precompute the class probability ratio differences (a CxC matrix)
            w = numpy.zeros((self.n_classes, self.n_classes))
            for c in range(self.n_classes):
                for chi in range(self.n_classes):
                    if c != chi:
                        w[c, chi] = (class_probability[chi] / class_probability[c]) - (
                            self.training_class_probability[chi]
                            / self.training_class_probability[c]
                        )

            # Evaluate the sum in the denominator for each class
            denom_sum = []
            for c in range(self.n_classes):
                denom_sum.append(numpy.sum(w[c, :] * density_ratios, axis=1))

            # Output has N rows to cover each example in 'scores' and C columns for each of the class
            denom_sum = numpy.asarray(denom_sum).transpose()

            # Assemble the rest of the denominator
            denom = 1 + self.training_class_probability.reshape(1, -1) * denom_sum

            # Compute calibrated probability: output has N rows to cover each example in 'scores' and C columns for the classes
            posterior_probability = (
                density_ratios * self.training_class_probability.reshape(1, -1) / denom
            )
        else:
            raise ValueError("Internal inconsistency")

        # Return result with matching shape to input
        if scalar_output:
            return float(posterior_probability[:, 1].flatten()[0])
        elif slice_class_1_output:
            return posterior_probability[:, 1].flatten()
        elif flatten_output:
            return posterior_probability.flatten()
        else:
            return posterior_probability
