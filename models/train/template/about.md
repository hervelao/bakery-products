
# sources

[gcp day 5 correction](https://github.com/lewagon/taxi-fare) contains a working solution with a different file architecture

[gcp day 4 notebook instructions](https://github.com/lewagon/data-challenges/blob/master/05-Production/04-Deploy-to-Production/Challenge/04-Deploy-to-Production-Challenge.ipynb) contains informations required in order to setup the gcp account

# usage

1. conf: fill all the variables in `Model/conf.py`
2. train: fill `Model/trainer.py` with the code used in order to train your model
3. predict: fill `Model/predict.py` with the code used in order to predict with your model
4. if you feel the need to change the name of the `Model` package, please update:
- `CONF_FILE` in `Makefile`
- `from Model.conf` in `setup.py`, `trainer.py` and `predict.py`
- `PACKAGE_NAME` in `conf.py`

# how does this work?

`Makefile` allows you to launch several tasks (listed below) in order to setup your local environment and train or predict locally or on gcp

`setup.py` stores the project variables used by gcp in order to setup the gcp environement

`req.py` is used by the `pip_install_reqs` task in order to extract `REQUIRED_PACKAGES` from `Model/conf.py` (using a single file limits the risk that the environment is not the same locally and on gcp)

# tasks

## show project conf

``` zsh
make                                    # lists all project variables
make variables                          # lists all project variables
```

## install local requirements

``` zsh
make pip_install_reqs                   # installs locally all requirements
```

## train locally

``` zsh
python Model/trainer.py                 # trains model locally
python -m Model.trainer                 # trains model locally
make run_locally                        # trains model locally
```

## predict locally

makefile task and code TODO

## train on gcp

``` zsh
make auth                               # logins to gcp
make set_project                        # sets project if for gcp
make gcp_submit_training                # trains model on gcp
```

## predict on gcp

makefile task TODO but the code is ok
