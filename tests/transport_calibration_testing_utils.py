import numpy
import sklearn
import sklearn.linear_model
import sklearn.datasets
import xgboost
from src import transport_calibration


def run_e2e(
    n_classes,
    model_type,
    ratio_estimator="logistic",
    n_train=20000,
    n_calibrate=3000,
    n_validate=10000,
):
    """Do an end-to-end test run on simulated data: simulate, fit a classifier, resample to a new target domain, calibrate, validate on some simple metrics

    n_classes -- number of classes to generate
    model_type -- type of model to use for the classifier, must be in ('logistic', 'xgboost')
    ratio_estimator -- parameter indicating which density estimator to use: 'histogram' only works for binary classification, 'logistic' for any dimensionality
    n_train -- number of samples to generate for training the classifier
    n_calibrate -- number of samples to generate for training the calibrator
    n_validate -- number of samples to generate for computing the validation statistics

    Note: to make this function simple, we do not give access to parameters to configure the dataset generator. The defaults should be fine
    for common values of n_classes and number of samples.

    """
    # Check inputs
    if model_type not in ("logistic", "xgboost"):
        raise ValueError(f"Invalid model_type = {model_type}")
    if ratio_estimator not in ("logistic", "histogram"):
        raise ValueError(f"Invalid ratio_estimator = {ratio_estimator}")
    if n_classes < 2:
        raise ValueError("Must have at least 2 classes")

    # Simulate data
    total_samples = n_train + n_calibrate + n_validate
    x, y = sklearn.datasets.make_classification(
        n_samples=total_samples,
        n_features=40,
        n_informative=10,
        class_sep=0.65,
        flip_y=0.25,
        n_classes=n_classes,
    )

    # Split the datasets
    x_train = x[0:n_train]
    y_train = y[0:n_train]
    x_calibrate = x[n_train:(n_train + n_calibrate)]
    y_calibrate = y[n_train:(n_train + n_calibrate)]
    x_validate = x[(n_train + n_calibrate):]
    y_validate = y[(n_train + n_calibrate):]

    # Fit the classifier
    if model_type == "logistic":
        model = sklearn.linear_model.LogisticRegression().fit(x_train, y_train)
    elif model_type == "xgboost":
        model = xgboost.XGBClassifier().fit(x_train, y_train)
    else:
        raise ValueError("Internal inconsistency")

    # Training domain had uniform class prevalence
    training_class_probability = numpy.asarray([1 / n_classes] * n_classes)

    # Compute model scores on the calibration dataset to use to train the calibrator
    y_calibrate_predicted = model.predict_proba(x_calibrate)

    # Fit the calibrator
    calibrator = transport_calibration.TransportCalibration(
        y_calibrate_predicted,
        y_calibrate,
        training_class_probability,
        ratio_estimator=ratio_estimator,
    )

    # For binary classification, the calibrator supports simplified input shapes, so repeat with simplified input shapes to test that logic
    if n_classes == 2:
        # Allocate storage for these additional test objects
        alt_objs = []

        # Test input shape of (N,)
        alt_objs.append(
            transport_calibration.TransportCalibration(
                y_calibrate_predicted[:, 1].flatten(),
                y_calibrate,
                training_class_probability,
                ratio_estimator=ratio_estimator,
            )
        )

        # Test input shape of (N,) and scalar training_class_probability
        alt_objs.append(
            transport_calibration.TransportCalibration(
                y_calibrate_predicted[:, 1].flatten(),
                y_calibrate,
                float(training_class_probability[1]),
                ratio_estimator=ratio_estimator,
            )
        )

        # Test input shape of (N,C) and scalar training_class_probability
        alt_objs.append(
            transport_calibration.TransportCalibration(
                y_calibrate_predicted,
                y_calibrate,
                float(training_class_probability[1]),
                ratio_estimator=ratio_estimator,
            )
        )

        # Test input shape of (N,C) and (1,) training_class_probability
        alt_objs.append(
            transport_calibration.TransportCalibration(
                y_calibrate_predicted,
                y_calibrate,
                numpy.asarray([float(training_class_probability[1])]),
                ratio_estimator=ratio_estimator,
            )
        )

    # Select a random class prevalence to use for the target domain
    rng = numpy.random.default_rng()
    expscale = 1 / 2.5
    alpha = float(rng.exponential(scale=expscale, size=1))
    class_probability = rng.dirichlet([alpha] * n_classes)

    # Resample the validation data to this target domain (factor of 4 may be adjusted if more or fewer samples are desired)
    target_domain_data = stratified_sample_on_label(
        class_probability, x_validate, y_validate, n_validate // 4
    )

    # Compute model scores on the resampled dataset
    y_validate_predicted = model.predict_proba(target_domain_data["x"])

    # Calibrate those scores for the target domain
    y_validate_predicted_calibrated = calibrator.calibrated_probability(
        y_validate_predicted, class_probability
    )

    # For binary classification, the calibrator supports simplified input shapes, so repeat with simplified input shapes to test that logic
    if n_classes == 2:
        # Allocate storate for additional calibration results
        alt_outputs = []

        # Loop over the calibrator objects to test each combination
        for cal_obj in [calibrator] + alt_objs:
            # Test input shape of (N,C)
            alt_outputs.append(
                cal_obj.calibrated_probability(y_validate_predicted, class_probability)
            )

            # Test input shape of (N,C) with scalar class_probability
            alt_outputs.append(
                cal_obj.calibrated_probability(
                    y_validate_predicted, float(class_probability[1])
                )
            )

            # Test input shape of (N,)
            alt_outputs.append(
                cal_obj.calibrated_probability(
                    y_validate_predicted[:, 1].flatten(), class_probability
                )
            )

            # Test input shape of (N,) with scalar class probability
            alt_outputs.append(
                cal_obj.calibrated_probability(
                    y_validate_predicted[:, 1].flatten(), float(class_probability[1])
                )
            )

    # Compute some validation metrics
    validation_metrics = {}
    validation_metrics["class_probability"] = class_probability
    validation_metrics["predicted_prevalence"] = y_validate_predicted.mean(axis=0)
    validation_metrics[
        "predicted_prevalence_calibrated"
    ] = y_validate_predicted_calibrated.mean(axis=0)

    # Compute validation metrics on the alternates for binary classification
    if n_classes == 2:
        for i in range(len(alt_outputs)):
            validation_metrics[f"alt_{i}"] = alt_outputs[i].mean(axis=0)
    return validation_metrics, calibrator


def stratified_sample_on_label(class_probability, x, y, n_samples):
    """Extract a subset of examples such that the prevalence of each class matches the specified class_probability

    class_probability -- array indicating the desired probability of each class (must sum to one)
    x -- array of features for each example
    y -- array of labels for each example
    n_samples -- number of desired output samples

    Returns a dictionary with the extracted 'x', 'y', and a value for the 'actual_p_event'

    Note: This function will fail if there are not a sufficient number of examples in the input data to produce
    an 'nsample' subset with the specified prevalence.

    """

    # Check that the input is normalized
    if numpy.sum(class_probability) > 1.1 or numpy.sum(class_probability) < 0.9:
        raise ValueError("Sum over all class probabilities must be one.")

    # Prepare output
    results = {}

    # Determine the number of examples of each class to extract from the test set
    nclassex = numpy.asarray(
        numpy.round(numpy.asarray(class_probability) * n_samples), dtype=int
    )

    # Prepare to track the actually realized class fractions as a sanity check
    actual_class_count = []

    # Select a random set of example indices for each class
    class_inds = []
    for c in range(len(class_probability)):
        # Find the indices and randomize their order
        temp_inds = numpy.where(y == c)[0]
        numpy.random.shuffle(temp_inds)

        # Select the desired quantity of examples
        try:
            class_inds.append(temp_inds[0:nclassex[c]])
        except IndexError:
            raise ValueError(
                f"Not enough examples of class {c} to extract a subset of size {n_samples}"
            )

        # Count the number of examples for tracking the actuals
        actual_class_count.append(len(temp_inds[0:nclassex[c]]))
    class_inds = numpy.concatenate(class_inds)

    # Measure the actually realized class fractions as a sanity check
    total_num = sum(actual_class_count)
    results["actual_class_probability"] = numpy.asarray(actual_class_count) / total_num

    # Extract those examples
    results["x"] = numpy.take(x, class_inds, axis=0)
    results["y"] = numpy.take(y, class_inds)
    return results
