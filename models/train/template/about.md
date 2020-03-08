
# sources

[gcp day 5 correction](https://github.com/lewagon/taxi-fare)
[day 4 notebook instructions](data-challenges/05-Production/04-Deploy-to-Production/Challenge/04-Deploy-to-Production-Challenge.ipynb)

# commands

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
