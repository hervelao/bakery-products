
# sources

[gcp day 5 correction](https://github.com/lewagon/taxi-fare)
[day 4 notebook instructions](data-challenges/05-Production/04-Deploy-to-Production/Challenge/04-Deploy-to-Production-Challenge.ipynb)

# requirements

``` zsh
pip install -r requirements.txt
```

# commands

train model locally or on gcp

both tasks use data from gcp bucket
and upload the trained dumped model to gcp bucket

## train locally

``` zsh
python StaticModel/trainer.py           # trains model localy
```

``` zsh
python -m StaticModel.trainer           # trains model localy
```

``` zsh
make run_locally                        # trains model locally
```

## train on gcp

``` zsh
make gcp_submit_training                # trains model on gcp
```
