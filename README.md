# Transport Calibration

[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml)
[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml)
[![codecov](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib/branch/main/graph/badge.svg?token=z6l3Vet7N6)](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib)

---

## Installation

```bash
pip install ds-transport-calibration
```

## Usage of XGBoost wrapper

This library provides a wrapper around [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier)
which allows seamless usage of transport calibration with the sklearn api and XGBoost.

The wrapper overloads predict_proba(...) so that it computes calibrated probabilities.

The wrapper is compatible with [shap](https://shap.readthedocs.io/en/latest/index.html), but the [tree explainer](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Tree.html) will use uncalibrated probabilities
because it computes the probability using a fast internal implementation that short-circuits predict_proba.
Alternatively, it is possible to use a universal explainer, like [permutation](https://shap.readthedocs.io/en/latest/generated/shap.explainers.Permutation.html), 
in order to compute shap values on calibrated scores.

```python
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

# Or, if the class probability is the same as the training domain then it can be automatically set by doing this instead
model.transport_calibration_class_probability = None

# Compute calibrated probabilities
calibrated_predictions = model.predict_proba(x_test)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on contributing to this project.
