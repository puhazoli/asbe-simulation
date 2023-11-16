def main():
    print("Loading packages")
    from asbe_simulation.helpers import (
        generate_data,
        CausalForestEstimator,
        PEHE,
        MajorityAssignmentFunction,
        dgp_x,
        dgp_t,
        dgp_y,
        prepare_data,
        XBARTEstimator,
        BARTEstimator,
        OPENBTITEEstimator,
        GPyEstimator,
        ExpectedReliability,
        create_table,
        plot_metric_al_functions,
        plot_metric_estimators,
    )

    from asbe.base import (
        BaseActiveLearner,
        BaseAcquisitionFunction,
        BaseAssignmentFunction,
        BaseITEEstimator,
        BaseDataGenerator,
    )
    from pyopenbt.openbt import OPENBT
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    import argparse

    parser = argparse.ArgumentParser(description='Run random variations with different acquisition functions')
    parser.add_argument('--num-variations', type=int, default=5, help='Number of random variations to run')
    parser.add_argument('--estimator', nargs='+', choices=['xbart', 'gpy', 'pybart', 'openbt'], default=['xbart', 'gpy', 'pybart'], help='Estimators to use [possible values are: "xbart", "gpy", "pybart", "openbt"]')
    parser.add_argument('--acq-function', nargs='+', choices=['random', 'emcm', 'unc', 'er'], default=['random', 'emcm', 'unc'], help='Acquisition functions to use [possible values are: "random", "emcm", "unc", "er"]')
    parser.add_argument('--num-steps', type=int, default=6, help='Number of steps to run the active learning algorithm')


    args = parser.parse_args()

    num_variations = args.num_variations
    estimators = args.estimator
    acq_function = args.acq_function
    num_steps = args.num_steps

    # Section 2.5 - Working example
    print("####### Section 2.5 #######")
    N = 1000
    ds = generate_data(N=N)  # generates dictionary of data, with X, t, y
    asl = BaseActiveLearner(
        estimator=BaseITEEstimator(model=LogisticRegression(), two_model=False),
        acquisition_function=BaseAcquisitionFunction(),
        assignment_function=BaseAssignmentFunction(),
        stopping_function=None,
        dataset=ds,
    )
    asl.fit()
    print(asl.dataset["X_pool"].shape)
    X_new, query_idx = asl.query(no_query=10)
    asl.teach(query_idx)
    print(f"PEHE: {asl.score():.2f}")
    print(asl.dataset["X_pool"].shape)
    print("####### Section 2.5 end #######")

    # Section 3.2
    # Needed to surpass numba warnings
    print("####### Section 3.2 #######")
    from numba.core.errors import NumbaDeprecationWarning
    import warnings

    warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
    from asbe.base import BaseActiveLearner
    from asbe.estimators import BaseITEEstimator
    from asbe.models import RandomAcquisitionFunction
    from sklearn.ensemble import RandomForestRegressor
    from econml.dml import CausalForestDML

    bite = BaseITEEstimator(
        model=RandomForestRegressor(), two_model=True, ps_model=None, dataset=ds
    )
    bite.fit()
    preds = bite.predict(X=ds["X_test"])
    print(preds.shape)

    # The result might vary due to randomness in the forest based algorithm

    bite_dml = CausalForestEstimator(
        model=CausalForestDML(), two_model=True, ps_model=None, dataset=ds
    )
    bite_dml.fit()
    preds = bite_dml.predict(X=ds["X_test"])
    print(np.mean(preds))
    print("####### Section 3.2 end #######")

    # Section 3.5
    print("####### Section 3.5 #######")
    ds = generate_data(N=N)  # generates dictionary of data, with X, t, y
    asl = BaseActiveLearner(
        estimator=BaseITEEstimator(model=LogisticRegression(), two_model=False),
        acquisition_function=BaseAcquisitionFunction(),
        assignment_function=BaseAssignmentFunction(),
        stopping_function=None,
        dataset=ds,
    )
    print(asl.simulate(no_query=1, metric=PEHE))
    print("####### Section 3.5 end #######")

    # Section 4.1
    print("####### Section 4.1 #######")

    # Reading in data and making train test split
    data = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        header=None,
    )
    col = [
        "treatment",
        "y_factual",
        "y_cfactual",
        "mu0",
        "mu1",
    ]
    for i in range(1, 26):
        col.append("x" + str(i))
    data.columns = col
    data = data.astype({"treatment": "bool"}, copy=False)
    X = data.loc[:, [col.startswith("x_") for col in data.columns]].to_numpy()
    y1 = np.where(data["treatment"], data.y_factual, data.y_cfactual)
    y0 = np.where(~data["treatment"], data.y_factual, data.y_cfactual)
    y = data.y_factual.to_numpy()
    t = data.treatment.to_numpy()
    ite = y1 - y0

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y1_train,
        y1_test,
        y0_train,
        y0_test,
        t_train,
        t_test,
        ite_train,
        ite_test,
    ) = train_test_split(X, y, y1, y0, t, ite)
    ds = {
        "X_training": X_train,
        "y_training": y_train,
        "t_training": t_train,
        "X_pool": deepcopy(X_test),
        "y_pool": deepcopy(y_test),
        "t_pool": deepcopy(t_test),
        "y1_pool": y1_test,
        "y0_pool": y0_test,
        "X_test": X_test,
        "y_test": y_test,
        "t_test": t_test,
        "ite_test": ite_test,
    }
    asl_ihdp = BaseActiveLearner(
        estimator=BaseITEEstimator(model=RandomForestRegressor(), two_model=False),
        acquisition_function=RandomAcquisitionFunction(),
        assignment_function=MajorityAssignmentFunction(),
        stopping_function=None,
        dataset=ds,
    )
    asl_ihdp.fit()
    print("####### Section 4.1 end #######")

    # Section 4.2
    print("####### Section 4.2 #######")
    l = BaseDataGenerator(ds=None, no_training=5, dgp_x=dgp_x, dgp_t=dgp_t, dgp_y=dgp_y)
    asl_rf = BaseActiveLearner(
        estimator=BaseITEEstimator(model=RandomForestRegressor(), two_model=False),
        acquisition_function=BaseAcquisitionFunction(),
        assignment_function=BaseAssignmentFunction(),
        stopping_function=None,
        dataset=l,
        offline=False,
    )
    _ = asl_rf.dataset.get_data(no_query=100, as_test=True)
    # Once again, there can be randomness in the result
    print(asl_rf.simulate(metric="Qini"))
    print("####### Section 4.2 end #######")

    # Section 5

    print("####### Section 5 #######")
    print("####### Main simulation #######")
    print("####### NOTE: This will take a while #######")
    print("Edit runner.py to change the estimators or acquistion functions")
    from asbe.models import (
        RandomAcquisitionFunction,
        EMCMAcquisitionFunction,
        UncertaintyAcquisitionFunction,
    )
    from xbart import XBART

    # In order to avoid unnecessary initialization of the models, we initialize them here
    ests = {}
    if "xbart" in estimators:
        ests["xbart"] = XBARTEstimator(
            name="xbart_optimized",
            model=XBART(num_sweeps=50, num_trees=400, beta=3, alpha=0.6),
        )
    if "gpy" in estimators:
        ests["gpy"] = GPyEstimator()
    if "pybart" in estimators:
        # NOTE: Possible unstable running
        ests["pybart"] = BARTEstimator(model=None)
    if "openbt" in estimators:
        # NOTE: This needs openmpi to be installed, brew install openmpi on mac, apt-get install openmpi on linux
        # NOTE: openbt is also slow and running this will significantly increase the runtime
        ests["openbt"] = OPENBTITEEstimator(model=OPENBT(model="bart"),two_model=True)
        
    acqs = {
        "random": RandomAcquisitionFunction(name="random", method="top"),
        "emcm": EMCMAcquisitionFunction(name="emcm", method="top"),
        "unc": UncertaintyAcquisitionFunction(name="unc"),
        "er": ExpectedReliability(name="er", method="top"),
    }
    # remove acquisition functions that are not in the list of acquisition functions
    acqs_to_run = [v for k, v in acqs.items() if k in acq_function]

    METRIC = "PEHE"
    model_results = {}
    for key, value in ests.items():
        res = {}
        for random_state in range(num_variations):
            for d in range(747):
                ds_main_simulation = prepare_data(d, random_state)
                asl_main_simulation = BaseActiveLearner(
                    estimator=value,
                    acquisition_function=acqs_to_run,
                    assignment_function=BaseAssignmentFunction(),
                    stopping_function=None,
                    dataset=ds_main_simulation,
                    al_steps=num_steps,
                )
                _ = asl_main_simulation.simulate(
                    no_query=1, metric=["PEHE", "decision"]
                )
                res[f"{key}_{random_state}_{d}"] = pd.DataFrame(
                    asl_main_simulation.simulation_results
                )
                res[f"{key}_{random_state}_{d}"]["sim"] = random_state
                res[f"{key}_{random_state}_{d}"]["data"] = d
        model_results[key] = create_table(res, METRIC)
        model_results[key]["model"] = key
    df_all = pd.concat(model_results.values())
    melted_df = df_all.melt(id_vars=["model", "sim", "data"])
    melted_df = melted_df[~melted_df["variable"].str.contains("change")]
    df = melted_df[["model", "data", "variable", "value"]]
    print(df)
    plot_metric_al_functions(df, metric=METRIC)
    plot_metric_estimators(df, metric=METRIC)
    print("####### Section 5 end #######")


if __name__ == "__main__":
    main()
