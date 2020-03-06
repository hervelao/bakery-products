from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'google-cloud-storage==1.26.0',
    'pandas==0.24.2',
    'scikit-learn==0.20.4']

PACKAGE_NAME='Model2Weeks'                        # model folder name
PACKAGE_DESCRIPTION='xgboost 2 weeks'

setup(name=PACKAGE_NAME,
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=PACKAGE_DESCRIPTION)
