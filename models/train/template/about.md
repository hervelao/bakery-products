
# sources

[gcp day 5 correction](https://github.com/lewagon/taxi-fare)
[day 4 notebook instructions](data-challenges/05-Production/04-Deploy-to-Production/Challenge/04-Deploy-to-Production-Challenge.ipynb)

# how does this work?

`.gitignore` lists files excluded from git

`about.md` is this file

`Makefile` allows you to launch several tasks in order to setup your local environment and train locally and on gcp

`setup.py` is used by gcp in order to setup the gcp environement

`req.py` is used by the `pip_install_reqs` task in order to extract `REQUIRED_PACKAGES` from `setup.py` (using a single files limits the risk that the environment is not the same locally and on gcp)

# tasks

train model locally or on gcp

both tasks use data from gcp bucket
and upload the trained dumped model to gcp bucket

## project setup

``` zsh
make                                    # lists all project variables
make variables                          # lists all project variables
```

## requirements

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
