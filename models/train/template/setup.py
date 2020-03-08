from setuptools import find_packages
from setuptools import setup

#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \
# `-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'
#
# this file is used by gcp and also provides parameters to Makefile
#

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# python model package
#

# folder name
PACKAGE_NAME="StaticModel"
PACKAGE_DESCRIPTION="static prediction model"

# file name
FILENAME="trainer"

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# gcp project
#

# project id
PROJECT_ID="le-wagon-data-grupo-bimbo"

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# gcp ai platform
#

# model training conf
REGION="europe-west1"
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
FRAMEWORK="scikit-learn"

REQUIRED_PACKAGES=[
    'google-cloud-storage==1.26.0',
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'scipy==1.2.2',
    'scikit-learn==0.20.4']

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# gcp storage
#

# model training bucket
BUCKET_NAME="wagon-data-grupo-bimbo-sales"

# model training folder
JOB_FOLDER="trainings"

# job prefix
JOB_PREFIX="static_model"

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# ai platform setup
#

setup(name=PACKAGE_NAME,
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=PACKAGE_DESCRIPTION)
