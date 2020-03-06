# do not use inline comments at the end of the lines,
# they add additional spaces in the content of the variables,
# which generates errors

# gcp project id
PROJECT_ID=le-wagon-data-grupo-bimbo
# gcp bucket name
BUCKET_NAME=wagon-data-grupo-bimbo-sales

# training folder
BUCKET_JOB_FOLDER=trainings
# training job name
JOB_NAME=static_model_$(shell date +'%Y%m%d_%H%M%S')

# package folder name
PACKAGE_NAME=StaticModel
# package file namme
FILENAME=trainer

# gcp training configuration
REGION=europe-west1
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
FRAMEWORK=scikit-learn

#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \
# `-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'
#
# my, my, is this not a nice comment
#

# auth to gcp
auth:
	-@gcloud auth login

# define project id
set_project:
	-@gcloud config set project ${PROJECT_ID}

# train model locally, from gcp bucket data, store trained model on gcp
run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

# train model on gcp, from gcp bucket data, store trained model on gcp
gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir "gs://${BUCKET_NAME}/${BUCKET_JOB_FOLDER}" \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs
