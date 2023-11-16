import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from asbe.base import BaseITEEstimator, BaseAcquisitionFunction, BaseAssignmentFunction
import pandas as pd
import pymc as pm
from asbe.base import *
from asbe.estimators import *
from scipy.stats import norm
from xbart import XBART
import pandas as pd
import pymc_bart as pmb
import pymc as pm
import GPy
import matplotlib.pyplot as plt


def generate_data(N, seed=1005):
    np.random.seed(seed)
    X = np.random.normal(size=N * 2).reshape((-1, 2))
    t = np.random.binomial(n=1, p=0.5, size=N)
    y = np.random.binomial(n=1, p=1 / (1 + np.exp(X[:, 1] * 2 + t * 3)))
    ite = 1 / (1 + np.exp(X[:, 1] * 2 + t * 3)) - 1 / (1 + np.exp(X[:, 1] * 2))
    (
        X_train,
        X_test,
        t_train,
        t_test,
        y_train,
        y_test,
        ite_train,
        ite_test,
    ) = train_test_split(X, t, y, ite, test_size=0.8, random_state=1005)
    ds = {
        "X_training": X_train,
        "y_training": y_train,
        "t_training": t_train,
        "X_pool": deepcopy(X_test),
        "y_pool": deepcopy(y_test),
        "t_pool": deepcopy(t_test),
        "X_test": X_test,
        "y_test": y_test,
        "t_test": t_test,
        "ite_test": ite_test,
    }
    return ds


class CausalForestEstimator(BaseITEEstimator):
    def fit(self, **kwargs):
        self.model.fit(
            Y=kwargs["y_training"], T=kwargs["t_training"], X=kwargs["X_training"]
        )

    def predict(self, **kwargs):
        return self.model.effect(kwargs["X"])


class UncertaintyAcquisitionFunction(BaseAcquisitionFunction):
    """Class to calculate variance in treatment effects"""

    def calculate_metrics(self, model, dataset):
        preds = model.predict(dataset["X_pool"])
        if preds.shape[0] <= 1 or len(preds["y1_preds"].shape) <= 1:
            raise Exception(
                "Not possible to calculate uncertainty when dimensions are <=1"
            )
        else:
            return np.var(preds, axis=1)


class MajorityAssignmentFunction(BaseAssignmentFunction):
    def select_treatment(self, model, dataset, query_idx):
        if sum(dataset["t_training"]) >= dataset["t_training"].shape[0] / 2:
            out = np.zeros((query_idx.shape[0],))
        else:
            out = np.ones((query_idx.shape[0],))
        return out


def PEHE(predictions, dataset):
    return np.sqrt(np.mean(np.square(predictions - dataset["ite_test"])))


def dgp_x(no_query=1):
    X1_10 = np.random.normal(size=(no_query, 10))
    X10_20 = np.random.binomial(1, 0.5, size=(no_query, 10))
    X = np.concatenate((X1_10, X10_20), axis=1)
    return X


def dgp_t(X):
    return np.random.binomial(1, 0.5, size=(X.shape[0]))


def dgp_y(X, t):
    y0 = 2.455 - (0.4 * X[:, 1] + 0.154 * X[:, 2] - 0.152 * X[:, 11] - 0.126 * X[:, 12])
    gx = (
        0.254 * X[:, 2] ** 2 - 0.152 * X[:, 11] - 0.4 * X[:, 11] ** 2 - 0.126 * X[:, 12]
    )
    tau = (
        0.4 * X[:, 1] + 0.154 * X[:, 2] - 0.152 * X[:, 11] - 0.126 * X[:, 12]
    ) - np.where(gx > 0, 1, 0)
    y1 = y0 + tau
    return np.where(t == 1, y1, y0)


class GPyEstimator(BaseITEEstimator):
    # https://github.com/IirisSundin/active-learning-for-decision-making/blob/master/src/gpmodel.py
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.name = "gpy"
        self.update = False

    def _create_model(self, x, y):
        d = x.shape[1]
        prior = GPy.core.parameterization.priors.Gamma(a=1.5, b=3.0)
        kern = GPy.kern.RBF(input_dim=d, ARD=True)
        kern.variance.set_prior(prior, warning=False)
        kern.lengthscale.set_prior(prior, warning=False)
        lik1 = GPy.likelihoods.Gaussian()
        lik1.variance.set_prior(prior, warning=False)
        lik_expert = GPy.likelihoods.Gaussian()
        lik_expert.variance.set_prior(prior, warning=False)
        lik = GPy.likelihoods.MixedNoise([lik1, lik_expert])
        output_index = np.ones(x.shape[0])
        model = GPy.core.GP(
            X=x,
            Y=y.reshape(-1, 1),
            kernel=kern,
            likelihood=lik,
            Y_metadata={"output_index": output_index},
        )
        model.optimize()
        return model

    def fit(self, **kwargs):
        if self.update:
            action = "t" if kwargs["t_training"] == 1 else "c"
            predictor = kwargs["X_training"]
            oracle_outcome = kwargs["y_training"]
            mcopy = {}
            mcopy["c"] = deepcopy(self.models["c"])
            mcopy["t"] = deepcopy(self.models["t"])
            mcopy[action].Y_metadata["output_index"] = np.r_[
                mcopy[action].Y_metadata["output_index"], np.array([1])
            ]
            mcopy[action].set_XY(
                np.r_[mcopy[action].X, predictor],
                np.r_[mcopy[action].Y, oracle_outcome],
            )
            mcopy[action].optimize()
            self.updated_model = mcopy
        else:
            self.models = {}
            Xt, yt = (
                kwargs["X_training"][(kwargs["t_training"] == 1), :],
                kwargs["y_training"][(kwargs["t_training"] == 1)],
            )
            Xc, yc = (
                kwargs["X_training"][(kwargs["t_training"] == 0), :],
                kwargs["y_training"][(kwargs["t_training"] == 0)],
            )
            self.models["c"] = self._create_model(Xc, yc)
            self.models["t"] = self._create_model(Xt, yt)

    def predict(self, X, **kwargs):
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        p1 = self.models["t"].posterior_samples_f(X, 100)
        p0 = self.models["c"].posterior_samples_f(X, 100)
        ite = p1 - p0
        ite = ite.squeeze(1)
        if "return_mean" in kwargs:
            if kwargs["return_mean"]:
                ite = (
                    self.models["t"].predict_noiseless(X)[0]
                    - self.models["c"].predict_noiseless(X)[0]
                )
        if "return_counterfactuals" in kwargs:
            if kwargs["return_counterfactuals"]:
                p1, p1s = self.models["t"]._raw_predict(X)
                p0, p0s = self.models["c"]._raw_predict(X)
                ite = (ite, p1, p0, p1s, p0s)
        return ite


def categorical2indicator(data, name, categorical_max=4):
    """
    Transforms categorical variable with name 'name' form a data frame to indicator variables

    Taken from https://github.com/IirisSundin/active-learning-for-decision-making/blob/e0c83f58181f81da2f867da4c49f1333fa7d0ae6/src/util.py#L14
    """
    values = data[name].values
    values[values >= categorical_max] = categorical_max
    uni = np.unique(values)
    for i, value in enumerate(uni):
        data[name + "." + str(i)] = np.array((values == value), dtype=int)
    data.drop(name, axis=1)
    return data


def preprocess(data, categorical_max=2):
    """
    This function preprocesses the hill data
    """
    # Normalization is enough here
    data["bw"] = normalize(data["bw"])
    data["nnhealth"] = normalize(data["nnhealth"])
    data["preterm"] = normalize(data["preterm"])
    # Taking logarithm does not harm here before normalizing, but might be unneccessary
    data["b.head"] = normalize(np.log(data["b.head"]))
    data["momage"] = normalize(np.log(data["momage"]))

    # Categorigal variables are made to indicators, could also be just normalized:
    # Birth order is between 1
    data = categorical2indicator(data, name="birth.o", categorical_max=categorical_max)

    # Everything else is binary, so processing doesn't really help, makes only understanding the results harder.

    # For some reason, indicator variable "first" is either 1 or 2, that is why we subtract 1 from it
    data["first"] = data["first"] - 1

    return data


def normalize(data):
    """
    Transforms the data to zero mean unit variance
    """
    return (data - np.mean(data)) / np.std(data)


def prepare_data(id_test, random_state, percent_for_train=0.1335):
    inputs = pd.read_csv(
        "https://raw.githubusercontent.com/puhazoli/ihdpdata/main/inputs.csv"
    )
    # names = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f'x{x}' for x in range(25)])
    inputs = preprocess(inputs, categorical_max=2)
    N = inputs.shape[0]
    N_train = int(np.ceil(N * percent_for_train))
    D = inputs.columns.shape[0]
    outcomes = pd.read_csv(
        "https://raw.githubusercontent.com/puhazoli/ihdpdata/main/observed_outcomes.csv",
        sep=",",
    )
    potential_outcomes = pd.read_csv(
        "https://raw.githubusercontent.com/puhazoli/ihdpdata/main/potential_outcomes.csv",
        sep=",",
    )
    potential_outcomes = potential_outcomes[["outcome_c0", "outcome_c1"]]
    actions = inputs["treat"]
    pred_names = inputs.columns[1:D]
    predictors = inputs[pred_names]

    outcomes = outcomes.values
    potential_outcomes = potential_outcomes.values
    actions = actions.values
    predictors = predictors.values

    outcomes = np.zeros(actions.shape)
    for i in range(actions.shape[0]):
        outcomes[i] = potential_outcomes[i, actions[i]]

    np.random.seed(random_state)
    ind = np.arange(N)
    np.random.shuffle(ind)
    outcomes = outcomes[ind]
    potential_outcomes = potential_outcomes[ind, :]
    actions = actions[ind]
    predictors = predictors[ind, :]

    # Creating training data
    X_test = np.copy(predictors[id_test, :])
    y_test = np.copy(outcomes[id_test])
    t_test = np.copy(actions[id_test])
    ite_test = potential_outcomes[id_test, 1] - potential_outcomes[id_test, 0]

    # shuffle the rest again (not test sample)
    np.random.seed(random_state)

    ind = np.arange(N - 1)
    np.random.shuffle(ind)
    outcomes = outcomes[np.arange(N) != id_test][ind]
    potential_outcomes = potential_outcomes[np.arange(N) != id_test][ind, :]
    actions = actions[np.arange(N) != id_test][ind]
    predictors = predictors[np.arange(N) != id_test, :][ind, :]

    X_train, y_train, t_train = (
        predictors[:N_train, :],
        outcomes[:N_train],
        actions[:N_train],
    )
    y0_train, y1_train = (
        potential_outcomes[:N_train, 0],
        potential_outcomes[:N_train, 1],
    )
    ite_train = y1_train - y0_train

    X_pool, y_pool, t_pool = (
        predictors[N_train:, :],
        outcomes[N_train:],
        actions[N_train:],
    )
    y0_pool, y1_pool = potential_outcomes[N_train:, 0], potential_outcomes[N_train:, 1]
    ite_pool = y1_pool - y0_pool

    ds = {
        "X_training": X_train,
        "y_training": y_train,
        "t_training": t_train,
        "ite_training": ite_train,
        "X_pool": X_pool,
        "y_pool": y_pool,
        "t_pool": t_pool,
        "y1_pool": y1_pool,
        "y0_pool": y0_pool,
        "ite_pool": ite_pool,
        "X_test": X_test.reshape((1, -1)),
        "y_test": y_test,
        "t_test": t_test,
        "ite_test": ite_test.reshape((-1, 1)),
    }
    return ds


class ExpectedReliability(BaseAcquisitionFunction):  #'decerr'
    """
    Uses Gauss-Hermite quadrature to compute expected type S error rate
    """

    def error_rate(self, preds):
        mu_tau = np.mean(preds[0])
        sd_tau = np.std(preds[0])
        alpha = norm.cdf(-np.abs(mu_tau) / sd_tau)
        return alpha

    def calculate_metrics(self, model, dataset):
        utilities = np.zeros(dataset["X_pool"].shape[0])
        reload = False
        for n in range(dataset["X_pool"].shape[0]):
            x_star = dataset["X_pool"][n, :].reshape(1, -1)
            a_star = 1 - dataset["t_pool"][n]  # counterfactual action
            decerr = 0.0
            points, weights = np.polynomial.hermite.hermgauss(16)
            for ii, yy in enumerate(points):
                preds_star = model.predict(X=x_star, return_counterfactuals=True)
                if a_star == 1:
                    mu_star, S_star = preds_star[1], preds_star[3]
                else:
                    mu_star, S_star = preds_star[2], preds_star[4]
                y_star = np.sqrt(2) * np.sqrt(S_star) * yy + mu_star  # for substitution
                new_data = {
                    "X_training": np.concatenate((dataset["X_training"], x_star)),
                    "y_training": np.concatenate(
                        (dataset["y_training"], y_star.reshape((1,)))
                    ),
                    "t_training": np.concatenate(
                        (dataset["t_training"], a_star.reshape((1,)))
                    ),
                }
                if model.name == "gpy":
                    model.update = True
                    model.fit(
                        X_training=x_star,
                        y_training=y_star,
                        t_training=a_star.reshape((1,)),
                    )
                    gpy_new = GPyEstimator()
                    gpy_new.models = model.updated_model
                    preds_next = gpy_new.predict(
                        X=dataset["X_test"], return_counterfactuals=True
                    )
                    model.update = False
                else:
                    try:
                        new_model = deepcopy(model)
                        new_model.fit(**new_data)
                        preds_next = new_model.predict(
                            X=dataset["X_test"], return_counterfactuals=True
                        )
                    except TypeError:
                        nm = XBARTEstimator(model=XBART(num_sweeps=20), two_model=False)
                        nm.fit(**new_data)
                        preds_next = nm.predict(
                            X=dataset["X_test"], return_counterfactuals=True
                        )
                util = 1 - self.error_rate(preds_next)
                decerr += weights[ii] * 1 / np.sqrt(np.pi) * util
            utilities[n] = decerr
        return decerr


class XBARTEstimator(BaseITEEstimator):
    def __init__(self, model, two_model=False, name=None):
        super().__init__(model=model, two_model=two_model, dataset=None, ps_model=None)
        self.name = "xbart" if name is None else name

    def predict(self, **kwargs):
        X0 = np.concatenate(
            (kwargs["X"], np.zeros(kwargs["X"].shape[0]).reshape((-1, 1))), axis=1
        )
        X1 = np.concatenate(
            (kwargs["X"], np.ones(kwargs["X"].shape[0]).reshape((-1, 1))), axis=1
        )
        if "return_mean" in kwargs:
            if kwargs["return_mean"] is True:
                out = self.model.predict(X1) - self.model.predict(X0)
        else:
            out = self.model.predict(X1, return_mean=False) - self.model.predict(
                X0, return_mean=False
            )
        if "return_counterfactuals" in kwargs:
            if kwargs["return_counterfactuals"] is True:
                p1 = self.model.predict(X1, return_mean=False)
                p0 = self.model.predict(X0, return_mean=False)
                return (out, np.mean(p1), np.mean(p0), np.std(p1), np.std(p0))
        return out


class BARTEstimator(BaseITEEstimator):
    def fit(self, **kwargs):
        self.model_bart = pm.Model()
        X = np.concatenate(
            (kwargs["X_training"], kwargs["t_training"].reshape((-1, 1))), axis=1
        )
        with self.model_bart:
            x_obs = pm.MutableData("X", X)
            self.model_bart.named_vars.pop("X")
            x_obs = pm.MutableData("X", X)
            μ = pmb.BART("μ", X=x_obs, Y=kwargs["y_training"], m=10)
            y_pred = pm.Normal(
                "y_pred", mu=μ, observed=kwargs["y_training"], shape=μ.shape
            )
            self.trace = pm.sample(random_seed=1005)

    def predict(self, **kwargs):
        X0 = np.concatenate(
            (kwargs["X"], np.zeros(kwargs["X"].shape[0]).reshape((-1, 1))), axis=1
        )
        X1 = np.concatenate(
            (kwargs["X"], np.ones(kwargs["X"].shape[0]).reshape((-1, 1))), axis=1
        )
        with self.model_bart:
            pm.set_data({"X": X1})
            p1 = pm.sample_posterior_predictive(self.trace)
            pm.set_data({"X": X0})
            p0 = pm.sample_posterior_predictive(self.trace)
            #             ite = p1["posterior_predictive"]["y_pred"] - p0["posterior_predictive"]["y_pred"]
            p1_ypred = p1["posterior_predictive"]["y_pred"].mean(axis=0)
            p0_ypred = p0["posterior_predictive"]["y_pred"].mean(axis=0)
            ite = p1_ypred - p0_ypred
            if "return_mean" in kwargs:
                if kwargs["return_mean"]:
                    out = ite.mean(axis=0).to_numpy().mean(axis=0)
            else:
                out = ite.T
            if "return_counterfactuals" in kwargs:
                if kwargs["return_counterfactuals"]:
                    out = (
                        ite.mean(axis=0),
                        p1_ypred.mean().values,
                        p1_ypred.std().values,
                        p0_ypred.mean().values,
                        p0_ypred.std().values,
                    )
        return out


def _normalize(df, col, max_number):
    if df.shape[0] == 1 or type(df) == pd.Series:
        out = pd.json_normalize(df.loc[f"{col}_1"]).rename(
            {i: f"{col}_step_{i}" for i in range(1, max_number+1)}, axis=1
        )
    else:
        out = pd.json_normalize(df.loc[:, f"{col}_1"]).rename(
            {i: f"{col}_step_{i}" for i in range(max_number+1)}, axis=1
        )
    return out


def create_table(res, metric="PEHE"):
    concated = pd.concat([x for x in res.values()])
    concated = concated.loc[metric]
    if concated.shape[0] == 1 or type(concated) == pd.Series:
        cols_to_check = concated.index
        sim_data_cols = [concated[["sim", "data"]].to_frame().T.reset_index(drop=True)]
    else:
        cols_to_check = concated.columns
        sim_data_cols = [concated[["sim", "data"]].reset_index(drop=True)]

    acq_function = [
        col.split("_")[0] for col in cols_to_check if col not in ["sim", "data"]
    ]
    # Get the max number out of the columns
    max_number = max(list(concated[f"{acq_function[0]}_1"].keys()))
    df_all = pd.concat(
        [_normalize(concated, acq, max_number) for acq in acq_function] + sim_data_cols,
        axis=1,
    )
    for col in acq_function:
        df_all[f"{col}_change"] = df_all[f"{col}_step_1"] - df_all[f"{col}_step_{max_number}"]
    return df_all


def plot_metric_al_functions(df, metric):
    # Extract information from the data
    acquisition_functions = set()
    active_learning_steps = set()
    values = {}

    for _, row in df.iterrows():
        data_id, variable, value = row["data"], row["variable"], row["value"]
        acquisition_function, step = variable.split("_step_")
        acquisition_functions.add(acquisition_function)
        active_learning_steps.add(int(step))

        if acquisition_function not in values:
            values[acquisition_function] = {"steps": [], "scores": []}

        values[acquisition_function]["steps"].append(int(step))
        values[acquisition_function]["scores"].append(value)

    # Sort the active learning steps in ascending order
    active_learning_steps = sorted(active_learning_steps)

    # Calculate mean and standard error per acquisition function at each step
    mean_scores = {}
    std_errors = {}

    for acquisition_function in acquisition_functions:
        mean_scores[acquisition_function] = []
        std_errors[acquisition_function] = []

        for step in active_learning_steps:
            scores = values[acquisition_function]["scores"]
            scores_at_step = [
                scores[i]
                for i in range(len(scores))
                if values[acquisition_function]["steps"][i] == step
            ]
            mean = np.mean(scores_at_step)
            std_error = np.std(scores_at_step) / np.sqrt(len(scores_at_step))
            mean_scores[acquisition_function].append(mean)
            std_errors[acquisition_function].append(std_error)

    # Plotting
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]  # Colors for different acquisition functions
    fig, ax = plt.subplots()

    custom_names = {
        "random": "Random",
        "er": "ExpectedReliability",
        "unc": "Uncertainty",
        "emcm": "BEMCMITE",
    }

    for i, acquisition_function in enumerate(acquisition_functions):
        steps = active_learning_steps
        scores = mean_scores[acquisition_function]
        std_err = std_errors[acquisition_function]
        color = colors[i % len(colors)]
        jittered_steps = np.array(steps) + np.random.uniform(-0.2, 0.2, size=1)
        #     jittered_scores = np.array(scores) + np.random.uniform(-0.03, 0.03, len(scores))
        ax.errorbar(
            jittered_steps,
            scores,
            yerr=std_err,
            marker="o",
            linestyle="-",
            color=color,
            label=custom_names.get(acquisition_function, acquisition_function),
        )

    ax.set_xticks(active_learning_steps)
    ax.set_xlabel("Active Learning Step")
    ax.set_ylabel(f"Mean {metric} score")
    ax.set_title("Performance of Different Active Learning Functions")
    ax.legend()
    plt.grid(True)
    plt.savefig(f"./figures/{metric}_acquisition.pdf")
    plt.show()


def plot_metric_estimators(df, metric):
    # Extract information from the DataFrame
    models = df["model"].unique()
    active_learning_steps = sorted(
        set(df["variable"].str.split("_step_").str[-1].astype(int))
    )

    # Calculate mean and standard error per model at each step
    mean_scores = {}
    std_errors = {}

    for model in models:
        mean_scores[model] = []
        std_errors[model] = []

        for step in active_learning_steps:
            scores = df.loc[df["model"] == model]
            scores_at_step = scores.loc[
                scores["variable"].str.endswith(f"step_{step}"), "value"
            ]
            mean = np.mean(scores_at_step)
            std_error = np.std(scores_at_step) / np.sqrt(len(scores_at_step))
            mean_scores[model].append(mean)
            std_errors[model].append(std_error)

    # Plotting
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    fig, ax = plt.subplots()

    for i, model in enumerate(models):
        scores = mean_scores[model]
        std_err = std_errors[model]
        color = colors[i % len(colors)]
        jittered_steps = np.array(active_learning_steps) + np.random.uniform(
            -0.1, 0.1, len(active_learning_steps)
        )
        jittered_scores = np.array(scores) + np.random.uniform(-0.03, 0.03, len(scores))
        ax.errorbar(
            jittered_steps,
            jittered_scores,
            yerr=std_err,
            marker="o",
            linestyle="-",
            color=color,
            label=model,
        )

    ax.set_xticks(active_learning_steps)
    ax.set_xlabel("Active Learning Step")
    ax.set_ylabel(f"{metric} score")
    ax.set_title(
        "Performance of Different Estimators with random selection at each step"
    )
    ax.legend()
    plt.grid(True)
    plt.savefig(f"./figures/{metric}_estimators.pdf")
    plt.show()
