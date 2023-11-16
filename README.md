# 0. Make sure the necessary dependencies are installed
- Python 3.9^
- openbtcli (from [here](https://bitbucket.org/mpratola/openbt/src/master/))

# 1. Install the project

`poetry install`
This will install all the packages with their intended versions

# 2. Run the project
`poetry run python runner.py`
This will run the single file that contains all the code found in the paper
If you'd like to modify the default acquisition functions and estimators, you can change:
`poetry run python runner.py --num-variation 1 --estimator xbart openbt --acq-function random unc`

The available arguments are:

--num-variations: This argument is used to specify the number of random variations to run. It is an integer argument with a default value of 5.

--estimator: This argument is used to specify the estimators to use. It is a list argument with possible values of "xbart", "gpy", "pybart", and "openbt". The default value is a list containing "xbart", "gpy", and "pybart".

--acq-function: This argument is used to specify the acquisition functions to use. It is a list argument with possible values of "random", "emcm", "unc", and "er". The default value is a list containing "random", "emcm", and "unc".

--num-steps: This argument is used to specify the number of steps to run the active learning algorithm. It is an integer argument with a default value of 6.