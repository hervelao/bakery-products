
# sources

[gcp day 5 correction](https://github.com/lewagon/taxi-fare)
[day 4 notebook instructions](data-challenges/05-Production/04-Deploy-to-Production/Challenge/04-Deploy-to-Production-Challenge.ipynb)

# usage

1. setup: fill all the variables in `setup.py`
2. train: fill YourModel/`trainer.py` with the code of your model
3. predict: fill YourModel/`predict.py`

# how does this work?

`.gitignore` lists files excluded from git

`about.md` is this file

`Makefile` allows you to launch several tasks (listed below) in order to setup your local environment and train locally or on gcp

`setup.py` stores the project variables (I could not figure out how to load them from a separate file and still have `setup.py` run correctly on gcp) and is used by gcp in order to setup the gcp environement

`req.py` is used by the `pip_install_reqs` task in order to extract `REQUIRED_PACKAGES` from `setup.py` (using a single files limits the risk that the environment is not the same locally and on gcp)

# tasks

## show project setup

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
python StaticModel/trainer.py           # trains model locally
python -m StaticModel.trainer           # trains model locally
make run_locally                        # trains model locally
```

## train on gcp

``` zsh
make auth                               # logins to gcp
make set_project                        # sets project if for gcp
make gcp_submit_training                # trains model on gcp
```
