# Transport Calibration

[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml)
[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml)
[![codecov](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib/branch/main/graph/badge.svg?token=z6l3Vet7N6)](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib)

---

## Summary

The transport calibration library allows a classifier to be calibrated post-hoc without modifying the already
trained model. But, unlike standard calibration methods, it avoids the need to fit a calibrator on data.

The simplest use-case is for a binary classifier where the positive class occurs
at a different rate in the real-world than it did in the training data. This commonly occurs when predicting
relatively rare events, because often there are more examples of the rare event in the training data than would
be encountered in the real world.

Rather than requiring an additional calibration dataset to train a post-hoc adjustment, as is done in the
standard calibrators built into
[sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV), transport calibration can "transport" the mis-calibrated model directly to
the desired base rate by simply inputting that base rate. For example, if a positive class occurred with a
frequency of 0.3 in the training data, but only occurs with a rate of 0.001 in the real world, these two
numbers are input into this library along with the already-trained classifier, and the calibration
will be immediately adjusted to produce positive-classes at the rate of 0.001.

This library also generalizes to multi-class classifiers, where the base rate is then input as a vector.

The above mentioned use-cases are all addressed by the TransportCalibration object provided by this library and the
corresponding XGB wrapper TransportCalibration_XGBClassifier.

A more unusual/advanced use-case is addressed by the TransportCalibrationOneCov. It applies when one particular
covariate (a.k.a. feature) depends on the domain and also affects the rate. For example, in predicting risk for
cancer patients, perhaps the occurrence-rate of the positive class depends on the age of the patient. In that case,
we could use patient-age as the adjustment covariate and then the pre-trained model could be calibrated with
a rate that depends on that feature. An XGB wrapper is also provided for this more advanced algorithm.

## Installation

```bash
pip install ds-transport-calibration
```

## Usage of XGBoost wrapper

This library provides a wrapper around [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
which allows seamless usage of transport calibration with the sklearn API and XGBoost.

The wrapper overloads predict_proba(...) so that it computes calibrated probabilities. And it provides access to the uncalibrated probabilities
via the method transport_calibration_predict_proba_uncalibrated(...)

The wrapper is compatible with [shap](https://shap.readthedocs.io/en/latest/index.html), but the [tree explainer](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Tree.html) will use uncalibrated probabilities
because it computes the probability using a fast internal implementation that short-circuits predict_proba.
Alternatively, it is possible to use a universal explainer, like [permutation](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html), 
in order to compute shap values on calibrated scores.

Here is example code to train a model, train the calibrator, and evaluate uncalibrated and calibrated probabilities.

```python
import numpy
import shap
import transport_calibration

# Fit the classifier
model = transport_calibration.TransportCalibration_XGBClassifier().fit(x_train, y_train)

# Define array with class prevalence in training domain
training_class_probability = numpy.asarray([1/n_classes]*n_classes)

# Fit the calibrator
model.transport_calibration_fit(x_calibrate, y_calibrate, training_class_probability=training_class_probability)

# Construct Shapley explainer: using fast tree method, but does not understand calibration, so shap values are on uncalibrated probability
tree_explainer = shap.TreeExplainer(model, data=x_calibrate[0:1000], model_output='predict_proba', feature_perturbation='interventional')

# Construct Shapley explainer: using generic method, slower, but shap values are on calibrated probability
perm_explainer = shap.PermutationExplainer(model.predict_proba, x_calibrate[0:100])

# Compute uncalibrated probabilities
uncalibrated_predictions = model.transport_calibration_predict_proba_uncalibrated(x_test)

# IMPORTANT: set the internal value of the class probability to the background value for the target domain before computing calibrated probabilities
model.transport_calibration_class_probability = class_probability

# Or, if the class probability is the same as the training domain then it can be automatically computed by setting this value to None
model.transport_calibration_class_probability = None

# Compute calibrated probabilities
calibrated_predictions = model.predict_proba(x_test)
```

### Fitting classifier and calibrator in a single call

In some situations it might be convenient to be able to construct a model
object with both the classifier and calibrator trained by a single call to the
fit method. This might be useful, for example, when using a cross-validation
method that repeatedly fits and tests the model.

To accomplish this, instantiate the model using an additional parameter to activate this auto-fitting mode.

```
model = transport_calibration.TransportCalibration_XGBClassifier(automatically_fit_calibrator_at_model_fit=True)
```

Then, a call to the model.fit(...) method will automatically fit the calibrator
with the same training data, after first fitting the classifier in the standard
way using that training data.

Note, this could produce unwanted results if the classifier-fit obliterates the
training data- meaning that the classifier's prediction on all the training
samples is nearly exactly 0 or 1 for the probability of the correct class. This
can happen on data that is very clean, where XGBoost is able to model the
training set with essentially no error.

## Usage of generic calibrator

For non-XGBoost applications, users may use the base calibrator object.

To train it, first compute probabilities from your model and store them in
an (N,C) numpy array (called model_scores here), where N is the number of examples and C is the number of classes. Then fit the calibrator by
```python
import transport_calibration

# Fit the calibrator
calibrator = transport_calibration.TransportCalibration(model_scores, y_labels, training_class_probability)
```

After the calibrator is fit, compute calibrated scores by
```python
# Compute calibrated probability
model_scores_calibrated = calibrator.calibrated_probability(model_scores, class_probability)
```

The class_probability is a numpy array containing the background class probability in the target domain.

## Advanced settings

Ratio estimator: when fitting a calibrator, it is possible to specify the internal algorithm to use for density estimates. This is
accomplished by passing ratio_estimator='logistic' or ratio_estimator='histogram' to the XGB wrapper's fit function or the generic
calibrator's constructor.
Typically, 'histogram' should be used for binary classification and 'logistic' should be used for multi-class.

Input/Output shapes: the generic calibrator object accepts a variety of input shapes for convenient usage. The output shape is determined so that it
is consistent with the input shape. See the docstring of the [calibrated_probability(...)](./src/transport_calibration/transport_calibration.py#L155) method for more details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on contributing to this project.
