import numpy
import sklearn
import sklearn.datasets
import shap
from tests import (
    transport_calibration_testing_utils,
    transport_calibration_one_cov_testing_utils,
)
from src import transport_calibration
from matplotlib import pyplot as plt


def run_xgb_wrapper_with_shap(
    ratio_estimator="logistic",
    n_train=20000,
    n_calibrate=3000,
    n_validate=10000,
    test_automatic_fit=False,
    make_pdfs=False,
):
    """Do an end-to-end test run on simulated data
    Simulate, fit a classifier, resample to a new target domain, calibrate, validate, compute SHAP

    ratio_estimator -- string indicating which density estimator to use:
                       'histogram' only works for binary classification
                       'logistic' for any dimensionality
    n_train -- number of samples to generate for training the classifier
    n_calibrate -- number of samples to generate for training the calibrator
    n_validate -- number of samples to generate for computing the validation statistics
    test_automatic_fit -- if True then test the automatic calibrator fitting functionality
    make_pdfs -- if True then write some example shap plots to PDF files

    """
    # Check inputs
    if ratio_estimator not in ("logistic", "histogram"):
        raise ValueError(f"Invalid ratio_estimator = {ratio_estimator}")

    # Define parameters that are hardcoded
    n_classes = 2
    n_resamples = 2

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
    x_calibrate = x[n_train : (n_train + n_calibrate)]
    y_calibrate = y[n_train : (n_train + n_calibrate)]
    x_validate = x[(n_train + n_calibrate) :]
    y_validate = y[(n_train + n_calibrate) :]

    # Select the fitting process
    if test_automatic_fit:
        # Fit the classifier and calibrator in a single call
        model = transport_calibration.TransportCalibration_XGBClassifier(
            automatically_fit_calibrator_at_model_fit=True
        ).fit(x_train, y_train)
    else:
        # Fit the classifier
        model = transport_calibration.TransportCalibration_XGBClassifier().fit(
            x_train, y_train
        )

        # Training domain had uniform class prevalence
        training_class_probability = numpy.asarray([1 / n_classes] * n_classes)

        # Fit the calibrator
        model.transport_calibration_fit(
            x_calibrate,
            y_calibrate,
            training_class_probability=training_class_probability,
            ratio_estimator=ratio_estimator,
        )

    # Construct Shapley explainer: using fast tree method, but shap values are on uncalibrated probability
    tree_explainer = shap.TreeExplainer(
        model,
        data=x_calibrate[0:1000],
        model_output="predict_proba",
        feature_perturbation="interventional",
    )

    # Construct Shapley explainer: using generic method, slower, but shap values are on calibrated probability
    perm_explainer = shap.PermutationExplainer(model.predict_proba, x_calibrate[0:100])

    # Define a couple probability vectors for different target domains
    class_probability = [numpy.asarray([0.2, 0.8]), numpy.asarray([0.7, 0.3])]

    # Define output storage
    target_domain_data = []
    y_validate_predicted = []
    y_validate_predicted_calibrated = []
    validation_metrics = {}

    # Compute validation across each target domain
    for i in range(n_resamples):
        # Resample the validation data to this target domain (factor of 4 may be adjusted if more or fewer samples are desired)
        target_domain_data.append(
            transport_calibration_testing_utils.stratified_sample_on_label(
                class_probability[i], x_validate, y_validate, n_validate // 4
            )
        )

        # Compute uncalibrated model scores on the resampled dataset
        y_validate_predicted.append(
            model.transport_calibration_predict_proba_uncalibrated(
                target_domain_data[i]["x"]
            )
        )

        # Compute calibrated model scores on the resampled dataset
        model.transport_calibration_class_probability = class_probability[i]
        y_validate_predicted_calibrated.append(
            model.predict_proba(target_domain_data[i]["x"])
        )

        # Compute some validation metrics
        validation_metrics[f"class_probability_{i}"] = class_probability[i]
        validation_metrics[f"predicted_prevalence_{i}"] = y_validate_predicted[i].mean(
            axis=0
        )
        validation_metrics[
            f"predicted_prevalence_calibrated_{i}"
        ] = y_validate_predicted_calibrated[i].mean(axis=0)

    # Plot Shapley values for several examples on each target domain and explainer algo
    sanity_check_count = 0
    for i in range(n_resamples):
        # Important to set the internal value of the class probability to the correct value for this domain
        model.transport_calibration_class_probability = class_probability[i]

        # Explain some rows with each explainers
        shaps = (
            tree_explainer(target_domain_data[i]["x"][0:3]),
            perm_explainer(target_domain_data[i]["x"][0:3]),
        )
        algos = ("tree", "permutation")

        # Compute fx from shap and also directly compute it as a sanity check
        fx_shap_unc = shaps[0].base_values[:, 1] + shaps[0][0:3, :, 1].values.sum(
            axis=1
        )
        fx_dir_unc = y_validate_predicted[i][0:3, 1]
        fx_shap_cal = shaps[1].base_values[:, 1] + shaps[1][0:3, :, 1].values.sum(
            axis=1
        )
        fx_dir_cal = y_validate_predicted_calibrated[i][0:3, 1]

        # These should match each other
        validation_metrics[f"sanitycheck_{sanity_check_count}"] = numpy.isclose(
            fx_shap_unc, fx_dir_unc
        ).all()
        sanity_check_count = sanity_check_count + 1
        validation_metrics[f"sanitycheck_{sanity_check_count}"] = numpy.isclose(
            fx_shap_cal, fx_dir_cal
        ).all()
        sanity_check_count = sanity_check_count + 1

        # Plot the explanation for each example
        if make_pdfs:
            for j in range(3):
                # Loop over algorithms
                for k in range(2):
                    shap.plots.waterfall(shaps[k][j][:, 1])
                    plt.title(
                        "{a}: calibrated pred={pred:.2f}, uncal={unc:.2f}".format(
                            a=algos[k],
                            pred=y_validate_predicted_calibrated[i][j, 1],
                            unc=y_validate_predicted[i][j, 1],
                        )
                    )
                    plt.savefig(f"waterfall_{algos[k]}_{j}_{i}.pdf")
                    plt.clf()
    return validation_metrics, model


def run_xgb_wrapper_with_shap_for_one_cov(
    n_train=20000,
    n_calibrate=3000,
    n_validate=10000,
    test_automatic_fit=False,
    predefined_domain_param_set="c3",
):
    """Do an end-to-end test run on simulated data
    Simulate, fit a classifier, resample to a new target domain, calibrate, validate, compute SHAP

    n_train -- number of samples to generate for training the classifier
    n_calibrate -- number of samples to generate for training the calibrator
    n_validate -- number of samples to generate for computing the validation statistics
    test_automatic_fit -- if True then test the automatic calibrator fitting functionality
    predefined_domain_param_set -- parameter set name for generating data

    """
    # Simulate data
    sd = transport_calibration_one_cov_testing_utils.generate_data(
        n_train, n_calibrate, n_validate, predefined_domain_param_set
    )

    # Select the fitting process
    if test_automatic_fit:
        # Fit the classifier and calibrator in a single call
        model = transport_calibration.TransportCalibrationOneCov_XGBClassifier(
            automatically_fit_calibrator_at_model_fit=True
        ).fit(sd["x_train"], sd["y_train"])

        # Using training data directly for the calibrator does not seem to work as well, so set a higher tolerance
        atol = 0.16

        # Set the target domain distribution using data from the primed domain rather than the training data
        model.set_primed_distribution(
            sd["yp_calibrate"],
            model.extract_one_cov_from_full_features(sd["xp_calibrate"]),
        )
    else:
        # Determine the index of the adjustment variable so that we can test the code that specifies it
        ocf_index = sd["x_train"].shape[1] - 1

        # Fit the classifier
        model = transport_calibration.TransportCalibrationOneCov_XGBClassifier(
            transport_calibration_one_cov_feature_index=ocf_index
        ).fit(sd["x_train"], sd["y_train"])

        # Fit the calibrator
        model.transport_calibration_fit(
            sd["x_calibrate"],
            sd["y_calibrate"],
            labels_primed=sd["yp_calibrate"],
            xvals_primed=model.extract_one_cov_from_full_features(sd["xp_calibrate"]),
        )
        atol = 0.06

    # Construct Shapley explainer: using fast tree method, but shap values are on uncalibrated probability
    tree_explainer = shap.TreeExplainer(
        model,
        data=sd["x_calibrate"][0:1000],
        model_output="predict_proba",
        feature_perturbation="interventional",
    )

    # Construct Shapley explainer: using generic method, slower, but shap values are on calibrated probability
    perm_explainer = shap.PermutationExplainer(
        model.predict_proba, sd["x_calibrate"][0:100]
    )

    # Define output storage
    validation_metrics = {}

    # Compute uncalibrated and calibrated model scores on the resampled dataset
    yp_validate_predicted_uncalibrated = (
        model.transport_calibration_predict_proba_uncalibrated(sd["xp_validate"])
    )
    yp_validate_predicted = model.predict_proba(sd["xp_validate"])

    # Compute some validation metrics
    validation_metrics[
        "predicted_prevalence_uncalibrated"
    ] = yp_validate_predicted_uncalibrated.mean(axis=0)
    validation_metrics["predicted_prevalence"] = yp_validate_predicted.mean(axis=0)
    validation_metrics["unprimed_actual_prevalence"] = sd["y_validate"].mean()
    validation_metrics["primed_actual_prevalence"] = sd["yp_validate"].mean()
    validation_metrics["sanity_check"] = numpy.isclose(
        validation_metrics["primed_actual_prevalence"],
        validation_metrics["predicted_prevalence"][1],
        atol=atol,
    )

    # Explain some rows with each explainers
    shaps = (
        tree_explainer(sd["xp_validate"][0:3]),
        perm_explainer(sd["xp_validate"][0:3]),
    )

    # Compute fx from shap and also directly compute it as a sanity check
    fx_shap_unc = shaps[0].base_values[:, 1] + shaps[0][0:3, :, 1].values.sum(axis=1)
    fx_dir_unc = yp_validate_predicted_uncalibrated[0:3, 1]
    fx_shap_cal = shaps[1].base_values[:, 1] + shaps[1][0:3, :, 1].values.sum(axis=1)
    fx_dir_cal = yp_validate_predicted[0:3, 1]

    # These should match each other
    sanity_check_count = 1
    validation_metrics[f"sanitycheck_{sanity_check_count}"] = numpy.isclose(
        fx_shap_unc, fx_dir_unc
    ).all()
    sanity_check_count = sanity_check_count + 1
    validation_metrics[f"sanitycheck_{sanity_check_count}"] = numpy.isclose(
        fx_shap_cal, fx_dir_cal
    ).all()
    sanity_check_count = sanity_check_count + 1
    return validation_metrics, model
