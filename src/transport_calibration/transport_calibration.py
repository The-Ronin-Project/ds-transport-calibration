import numpy
import sklearn
import sklearn.linear_model


class TransportCalibration():
    def __init__(self, raw_pred, labels, training_class_probability, ratio_estimator='logistic'):
        """ Initialize the calibration adjustment

        raw_pred -- numpy array containing the model-score vectors with shape (N,C) (N-number of examples, C-number of classes)
        labels -- numpy array of length N containing an integer from 0 to C-1 corresponding to the label for each row of 'raw_pred'
        training_class_probability -- a numpy array of length C containing the class probability in the training data
        ratio_estimator -- parameter indicating which density estimator to use: 'histogram' only works for binary classification, 'logistic' for any dimensionality

        Note: for binary classification, this class accepts simplified array shapes a follows:
            raw_pred may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)
            training_class_probability may be shape (1,) or it may be a scalar float. That value is assumed to correspond to C=1.

        """
        # Check inputs and reshape as necessary
        if isinstance(training_class_probability, numpy.ndarray) and len(training_class_probability.shape)==1 and training_class_probability.shape[0] == 1:
            # For binary classification this is allowed: temporarily replace the value so that it will get adjusted in the next check
            training_class_probability = float(training_class_probability[0])
        if isinstance(training_class_probability, float):
            # For binary classification this is allowed: adjust the value and shape to be consistent with a multi-class input
            training_class_probability = numpy.asarray([1-training_class_probability, training_class_probability])
        if not (isinstance(raw_pred, numpy.ndarray) and isinstance(labels, numpy.ndarray) and isinstance(training_class_probability, numpy.ndarray)):
            raise ValueError('Input values must be numpy arrays.')
        if len(raw_pred.shape) == 1 or raw_pred.shape[1] == 1:
            # For binary classification this is allowed: adjust the value and shape to be consistent with a multi-class input
            raw_pred = numpy.concatenate([1 - raw_pred.reshape(-1,1), raw_pred.reshape(-1,1)], axis=1)

        # Repair class_probability if it contains any exactly 0 or negative values
        training_class_probability = self._repair_class_probability(training_class_probability)

        # Store the prevalence of classes in training data
        self._training_class_probability = numpy.asarray(training_class_probability).flatten()

        # Store the number of classes
        self._n_classes = len(self.training_class_probability)

        # Store which type of estimator to use for the ratio of densities
        if ratio_estimator not in ('logistic', 'histogram'):
            raise ValueError(f'Invalid input for ratio_estimator parameter = {ratio_estimator}')
        if ratio_estimator == 'histogram' and self.n_classes != 2:
            raise ValueError('The histogram estimator only works for binary classification.')
        self._ratio_estimator = ratio_estimator

        # Construct an estimate of the ratio of distributions
        if self.ratio_estimator == 'logistic':
            # Construct a direct estimate of ratio of distributions for each class by fitting a Logistic regression
            self._logistic_model = sklearn.linear_model.LogisticRegression().fit(raw_pred, labels)
            self._ratios = lambda r, lr_model=self._logistic_model, tcp=self.training_class_probability: lr_model.predict_proba(r) / tcp
        elif self.ratio_estimator == 'histogram':
            # Construct initial version of histograms to estimate ratio of densities
            P_R = numpy.histogram(raw_pred[:,1].flatten(), bins='auto', density=False)
            P_R_Y1 = numpy.histogram(raw_pred[:,1].flatten()[numpy.where(labels==1)], bins=P_R[1], density=False)

            # Remove bins with zeros and recompute the histograms in order to avoid dividing by zero in the ratio
            zero_bins = numpy.where(P_R_Y1[0]==0)[0]
            nonzero_bins = numpy.delete(P_R_Y1[1], zero_bins)
            P_R = numpy.histogram(raw_pred[:,1].flatten(), bins=nonzero_bins, density=False)
            P_R_Y1 = numpy.histogram(raw_pred[:,1].flatten()[numpy.where(labels==1)], bins=nonzero_bins, density=False)

            # Store data needed for interpolating a value from the ratio of densities
            self._ratio_data = {}
            P_Y1 = self.training_class_probability[1]
            self._ratio_data['xp'] = P_R[1][0:-1]
            self._ratio_data['fp'] = P_Y1*P_R[0]/P_R_Y1[0]
            self._ratio_data['left'] = P_Y1*P_R[0][0]/P_R_Y1[0][0]
            self._ratio_data['right'] = P_Y1*P_R[0][-1]/P_R_Y1[0][-1]

            # Assemble histograms into a ratio interpolator
            self._ratio = lambda r, rd=self._ratio_data: numpy.interp(r, rd['xp'], rd['fp'], rd['left'], rd['right'])


    @property
    def n_classes(self):
        """ Number of classes that this calibrator was initialized for (immutable) """
        return self._n_classes


    @property
    def ratio_estimator(self):
        """ Type of ratio estimator that will be used to compute calibrated values (immutable) """
        return self._ratio_estimator


    @property
    def training_class_probability(self):
        """ The prior probability from the training domain of the classifier (immutable) """
        return self._training_class_probability


    def _repair_class_probability(self, class_probability):
        """ Repair the class probability so that it does not contain values less than or equal to zero

        class_probability -- a numpy array of length C containing the average class-probability

        """
        # Find offending values
        bad_inds = numpy.where(class_probability <= 0)[0]

        # Repair them
        if bad_inds.shape[0] != 0:
            # Set offending values to machine-precision small positive value > 0
            class_probability[bad_inds] = numpy.finfo(float).resolution
            print(f'Warning: class_probability had zeros, fixed it')

        # Renormalize if needed
        if class_probability.sum() != 1:
            class_probability = class_probability / class_probability.sum()
            print(f'Renormalized class probability = {class_probability}')
        return class_probability


    def calibrated_probability(self, scores, class_probability):
        """ Compute P'(Y=c | R) ie. the posterior probability, for a domain with prior class_probability, that Y is class c

        scores -- numpy array containing the model-scores to be adjusted with shape (N,C) (N-number of examples, C-number of classes)
        class_probability -- a numpy array of length C containing the average class-probability in the target domain

        Note: class_probability may not contain values less than or EQUAL to zero. These will be automatically adjusted if found.

        Note: for binary classification, this class accepts simplified array shapes a follows:
            scores may be shape (N,) and then the score is assumed to correspond to C=1 (the positive class)
            class_probability may be shape (1,) or it may be a scalar float. That value is assumed to correspond to C=1.


        Shape of the output depends on the shape of the input 'scores'

            Multi-class (n_classes >= 3):
                if scores.shape is (N,C) then output will have shape (N,C)      (corresponds to N examples)
                if scores.shape is (C,) then then output will have shape (C,)   (corresponds to a single example)

            Binary (n_classes == 2):
                if scores.shape is (N,2) then output will have shape (N,2)      (corresponds to multi-class case)

                if scores.shape is (N,) then output will have shape (N,), and each value will be the C=1 value
                if scores is float then the output will be float, and will be the C=1 value
        """
        # Define default output flags (these control the shape of the output according to the input, as discussed in the docstring)
        scalar_output = False
        flatten_output = False
        slice_class_1_output = False

        # Check inputs and reshape as necessary
        if self.n_classes == 2:
            # Identify invalid inputs
            if isinstance(class_probability, numpy.ndarray) and len(class_probability) > 2:
                raise ValueError('Trained for binary classification but got unexpected shape = {shape} for class probability.'.format(shape=class_probability.shape))
            if isinstance(scores, numpy.ndarray):
                if (len(scores.shape) == 2 and scores.shape[1] > 2) or len(scores.shape) > 2:
                    raise ValueError('Trained for binary classification but got unexpected shape = {shape} for input scores.'.format(shape=scores.shape))

            # Identify variations of allowed inputs
            if isinstance(class_probability, numpy.ndarray) and len(class_probability.shape)==1 and class_probability.shape[0] == 1:
                # For binary classification this is allowed: temporarily replace the value so that it will get adjusted in the next check
                class_probability = float(class_probability[0])
            if isinstance(class_probability, float):
                # For binary classification this is allowed: adjust the value and shape to be consistent with a multi-class input
                class_probability = numpy.asarray([1-class_probability, class_probability])
            if isinstance(scores, float):
                # This is the 'scores is float' case mentioned in the docstring
                scalar_output = True
                shaped_scores = numpy.asarray([1-scores, scores]).reshape(1,2)
        if not scalar_output and not (isinstance(scores, numpy.ndarray) and isinstance(class_probability, numpy.ndarray)):
            raise ValueError('Input values must be numpy arrays.')

        # Repair class_probability if it contains any exactly 0 or negative values
        class_probability = self._repair_class_probability(class_probability)

        # Determine the shape of the output according to the shape of the input and the type of model
        if self.n_classes == 2 and not scalar_output and len(scores.shape) == 1:
            # This is the scores.shape is (N,) case mentioned in the docstring
            shaped_scores = numpy.concatenate([1 - scores.reshape(-1,1), scores.reshape(-1,1)], axis=1)
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
        if self.ratio_estimator == 'histogram':
            # Evaluate the posterior probability: this ratio estimator only supports binary classifiers
            prior_ratio = (1 - class_probability[1]) / (1 - self.training_class_probability[1])
            posterior_probability = numpy.clip(class_probability[1] / (class_probability[1] + (self._ratio(shaped_scores[:,1].flatten())-self.training_class_probability[1])*prior_ratio), numpy.finfo(float).resolution, 1-numpy.finfo(float).resolution)

            # Reshape to (N,2) so that the output shape can be modified at end of this method via a general procedure
            posterior_probability = numpy.concatenate([1-posterior_probability.reshape(-1,1), posterior_probability.reshape(-1,1)], axis=1)
        elif self.ratio_estimator == 'logistic':
            # Precompute the density ratios: output has N rows for each of the N examples in 'scores' and C columns for each of the classes
            density_ratios = self._ratios(shaped_scores)

            # Precompute the class probability ratio differences (a CxC matrix)
            w = numpy.zeros((self.n_classes, self.n_classes))
            for c in range(self.n_classes):
                for chi in range(self.n_classes):
                    if c != chi:
                        w[c, chi] = (class_probability[chi] / class_probability[c]) - (self.training_class_probability[chi] / self.training_class_probability[c])

            # Evaluate the sum in the denominator for each of the classes: output has N rows for each of the N examples in 'scores' and C columns for each of the classes
            denom_sum = []
            for c in range(self.n_classes):
                denom_sum.append(numpy.sum(w[c,:]*density_ratios, axis=1))
            denom_sum = numpy.asarray(denom_sum).transpose()

            # Assemble the rest of the denominator
            denom = 1 + self.training_class_probability.reshape(1,-1)*denom_sum

            # Evaluate the posterior probability: output has N rows for each of the N examples in 'scores' and C columns for each of the classes
            posterior_probability = density_ratios*self.training_class_probability.reshape(1,-1) / denom
        else:
            raise ValueError('Internal inconsistency')

        # Return result with matching shape to input
        if scalar_output:
            return float(posterior_probability[:,1].flatten()[0])
        elif slice_class_1_output:
            return posterior_probability[:,1].flatten()
        elif flatten_output:
            return posterior_probability.flatten()
        else:
            return posterior_probability
