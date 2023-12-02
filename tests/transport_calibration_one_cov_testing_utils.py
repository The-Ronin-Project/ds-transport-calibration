import numpy
import scipy
import scipy.stats
import sklearn
import sklearn.linear_model
import sklearn.datasets
import xgboost
from src.transport_calibration import transport_calibration_one_cov
from tests.transport_calibration_testing_utils import stratified_sample_on_label


def run_e2e(  # noqa: C901
    n_classes=2,
    model_type="xgboost",
    n_train=20000,
    n_calibrate=3000,
    n_validate=10000,
    predefined_domain_param_set="c1",
):
    """End-to-end test run on simulated data
    Simulate, fit a classifier, resample to a new target domain, calibrate, validate on some simple metrics

    n_classes -- number of classes to generate
    model_type -- type of model to use for the classifier, must be in ('logistic', 'xgboost')
    n_train -- number of samples to generate for training the classifier
    n_calibrate -- number of samples to generate for training the calibrator
    n_validate -- number of samples to generate for computing the validation statistics
    predefined_domain_param_set -- a string indicating the name of a set of pre-defined parameters
                                   (see get_params_for_generating_domain_specific_feature(...) function below)

    Note: to make this function simple, there is no way to pass parameters to the simulator.
    The defaults should be fine for common values of n_classes and number of samples.

    """
    # Check inputs
    if model_type not in ("logistic", "xgboost"):
        raise ValueError(f"Invalid model_type = {model_type}")
    if n_classes != 2:
        raise ValueError("Only 2 classes are supported at the moment")

    # Simulate data
    sd = generate_data(
        n_train,
        n_calibrate,
        n_validate,
        predefined_domain_param_set,
        n_classes=n_classes,
    )

    # Fit the classifier
    if model_type == "logistic":
        model = sklearn.linear_model.LogisticRegression().fit(
            sd["x_train"], sd["y_train"]
        )
    elif model_type == "xgboost":
        model = xgboost.XGBClassifier().fit(sd["x_train"], sd["y_train"])
    else:
        raise ValueError("Internal inconsistency")

    # Select a random class prevalence to use for the target domain
    rng = numpy.random.default_rng()
    expscale = 1 / 2.5
    alpha = float(rng.exponential(scale=expscale, size=1))
    class_probability = None
    while class_probability is None:
        class_probability = rng.dirichlet([alpha] * n_classes)

        # Force a re-selection if any single value is too small
        if numpy.any(class_probability < 0.05):
            class_probability = None

    # Resample the data to achieve desired prevalence (factor of 4 may be adjusted if more or fewer samples are desired)
    target_domain_data_calibrate = stratified_sample_on_label(
        class_probability, sd["xp_calibrate"], sd["yp_calibrate"], n_calibrate // 4
    )
    target_domain_data_validate = stratified_sample_on_label(
        class_probability, sd["xp_validate"], sd["yp_validate"], n_validate // 4
    )

    # Compute model scores on the calibration dataset to use to train the calibrator
    y_calibrate_predicted = model.predict_proba(sd["x_calibrate"])

    # Fit the calibrator
    calibrator = transport_calibration_one_cov.TransportCalibrationOneCov(
        y_calibrate_predicted,
        sd["y_calibrate"],
        sd["x_calibrate"][:, -1].flatten(),
        labels_primed=target_domain_data_calibrate["y"],
        xvals_primed=target_domain_data_calibrate["x"][:, -1].flatten(),
    )

    # For binary classification, the calibrator supports simplified input shapes, so repeat tests with alternate shapes
    if n_classes == 2:
        # Allocate storage for these additional test objects
        alt_objs = []

        # Test input shape of (N,)
        alt_objs.append(
            transport_calibration_one_cov.TransportCalibrationOneCov(
                y_calibrate_predicted[:, 1].flatten(),
                sd["y_calibrate"],
                sd["x_calibrate"][:, -1].flatten(),
                labels_primed=target_domain_data_calibrate["y"],
                xvals_primed=target_domain_data_calibrate["x"][:, -1].flatten(),
            )
        )

    # Compute model scores on the resampled dataset
    y_validate_predicted = model.predict_proba(target_domain_data_validate["x"])

    # Calibrate those scores for the target domain
    xvals_primed = target_domain_data_validate["x"][:, -1].flatten()
    y_validate_predicted_calibrated = calibrator.calibrated_probability(
        y_validate_predicted, xvals_primed
    )

    # For binary classification, the calibrator supports simplified input shapes, so repeat tests with alternate shapes
    if n_classes == 2:
        # Allocate storate for additional calibration results
        alt_outputs = []

        # Loop over the calibrator objects to test each combination
        for cal_obj in [calibrator] + alt_objs:
            # Test input shape of (N,C)
            alt_outputs.append(
                cal_obj.calibrated_probability(y_validate_predicted, xvals_primed)
            )

            # Test input shape of (N,)
            alt_outputs.append(
                cal_obj.calibrated_probability(
                    y_validate_predicted[:, 1].flatten(), xvals_primed
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


def generate_data(
    n_train, n_calibrate, n_validate, predefined_domain_param_set, n_classes=2
):
    """Generate data using a simulation

    n_train -- number of samples to generate for training the classifier
    n_calibrate -- number of samples to generate for training the calibrator
    n_validate -- number of samples to generate for computing the validation statistics
    predefined_domain_param_set -- a string indicating the name of a set of pre-defined parameters
                                   (see get_params_for_generating_domain_specific_feature(...) function below)
    n_classes -- number of classes to generate

    Note: to make this function simple, there is no way to pass parameters to the simulator.
    The defaults should be fine for common values of n_classes and number of samples.

    """
    # Check inputs
    if n_classes != 2:
        raise ValueError("Only 2 classes are supported at the moment")

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

    # Modify simulated data to have 1 domain-specific covariate
    appended, appended_p = append_domain_specific_feature(
        x,
        y,
        get_params_for_generating_domain_specific_feature(predefined_domain_param_set),
    )

    # Split the datasets in the unprimed domain
    out = {}
    out["x_train"] = appended["x"][0:n_train]
    out["y_train"] = appended["y"][0:n_train]
    out["x_calibrate"] = appended["x"][n_train : (n_train + n_calibrate)]
    out["y_calibrate"] = appended["y"][n_train : (n_train + n_calibrate)]
    out["x_validate"] = appended["x"][(n_train + n_calibrate) :]
    out["y_validate"] = appended["y"][(n_train + n_calibrate) :]

    # Split the calibration/validation sets in the target (ie. primed) domain
    out["xp_calibrate"] = appended_p["x"][n_train : (n_train + n_calibrate)]
    out["yp_calibrate"] = appended_p["y"][n_train : (n_train + n_calibrate)]
    out["xp_validate"] = appended_p["x"][(n_train + n_calibrate) :]
    out["yp_validate"] = appended_p["y"][(n_train + n_calibrate) :]
    return out


def get_params_for_generating_domain_specific_feature(name):
    """Return predefined parameter settings

    name -- the name of a predefined parameter set

    """

    # Define parameter sets
    param_sets = {}
    param_sets["c1"] = {}
    param_sets["c1"]["dist"] = "normal"
    param_sets["c1"]["dist_params"] = (0.2, 1.3)
    param_sets["c1"]["dist_params_p"] = (0.5, 1.1)
    param_sets["c1"]["stick_prob_fn"] = lambda r: (r**1.7) / 2
    param_sets["c1"]["stick_prob_fn_p"] = lambda r: (r**1.3) / 2
    param_sets["c1"]["stick_up"] = True
    param_sets["c1"]["stick_up_p"] = False
    param_sets["c2"] = {}
    param_sets["c2"]["dist"] = "normal"
    param_sets["c2"]["dist_params"] = (0.2, 1.3)
    param_sets["c2"]["dist_params_p"] = (-0.2, 0.7)
    param_sets["c2"]["stick_prob_fn"] = lambda r: (r**1.3) / 2
    param_sets["c2"]["stick_prob_fn_p"] = lambda r: (r**1.7) / 2
    param_sets["c2"]["stick_up"] = True
    param_sets["c2"]["stick_up_p"] = False
    param_sets["c3"] = {}
    param_sets["c3"]["dist"] = "normal"
    param_sets["c3"]["dist_params"] = (0.2, 1.3)
    param_sets["c3"]["dist_params_p"] = (-0.5, 0.8)
    param_sets["c3"]["stick_prob_fn"] = lambda r: (r**1.7) / 2
    param_sets["c3"]["stick_prob_fn_p"] = lambda r: (r**1.3) / 2
    param_sets["c3"]["stick_up"] = False
    param_sets["c3"]["stick_up_p"] = True

    # Return desired set
    if name in param_sets.keys():
        return param_sets[name]
    else:
        raise ValueError(f"The predefined set {name} is not available")


def append_domain_specific_feature(x, y, params):
    """Append a feature 'xnew' to the set 'x' that causes 'y' to stick (ie. flip to 1) as a function of this new feature

    x -- array of domain agnostic features that predict 'y'
    y -- array of outcome labels consisting of 0 or 1
    params -- a dict of parameters describing the distributions needed to generate 'xnew' and to modify 'y'

    params contains keys:

    dist -- distribution of 'xnew': a string indicating distribution to draw from (must be 'normal' for now)
    dist_params -- tuple of parameters of the distribution
    dist_params_p -- tuple of parameters of the distribution in the primed domain
    stick_prob_fn -- function that takes a quantile from a continuous 'xnew' and returns sticking probability
    stick_prob_fn_p -- same as 'stick_prob_fn' but in primed domain
    stick_up -- indicator of whether 'y' should stick to 1 (if False then stick to 0)
    stick_up_p -- indicator of whether 'y' in primed domain should stick to 1 (if False then stick to 0)

    """

    # Generate values of x
    n = y.shape[0]
    if params["dist"] == "normal":
        # Construct the distribution functions
        distfn = scipy.stats.norm(
            loc=params["dist_params"][0], scale=params["dist_params"][1]
        )
        distfn_p = scipy.stats.norm(
            loc=params["dist_params_p"][0], scale=params["dist_params_p"][1]
        )

        # Generate values of 'xnew' in each domain
        xnew = distfn.rvs(size=n)
        xnew_p = distfn_p.rvs(size=n)

        # Evaluate the sticking probability for each value of xnew in each domain
        stick_prob = params["stick_prob_fn"](distfn.cdf(xnew))
        stick_prob_p = params["stick_prob_fn_p"](distfn_p.cdf(xnew_p))

    # Generate sticking indicators in each domain
    do_stick = numpy.random.binomial(1, stick_prob)
    do_stick_p = numpy.random.binomial(1, stick_prob_p)

    # Compute sticked version of y in each domain
    if params["stick_up"]:
        y_stick = (1 - do_stick) * y + do_stick
    else:
        y_stick = (1 - do_stick) * y
    if params["stick_up_p"]:
        y_stick_p = (1 - do_stick_p) * y + do_stick_p
    else:
        y_stick_p = (1 - do_stick_p) * y

    # Gather results
    appended = {"x": numpy.concatenate([x, xnew.reshape(-1, 1)], axis=1), "y": y_stick}
    appended_p = {
        "x": numpy.concatenate([x, xnew_p.reshape(-1, 1)], axis=1),
        "y": y_stick_p,
    }
    return appended, appended_p
