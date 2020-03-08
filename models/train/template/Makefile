# do not use inline comments at the end of the lines,
# they add additional spaces in the content of the variables,
# which generates errors

# conf file
CONF_FILE=.conf.py

# extract variables
PACKAGE_NAME=$(shell    grep "^PACKAGE_NAME="    ${CONF_FILE} | cut -d"=" -f2)
FILENAME=$(shell        grep "^FILENAME="        ${CONF_FILE} | cut -d"=" -f2)
PROJECT_ID=$(shell      grep "^PROJECT_ID="      ${CONF_FILE} | cut -d"=" -f2)
REGION=$(shell          grep "^REGION="          ${CONF_FILE} | cut -d"=" -f2)
PYTHON_VERSION=$(shell  grep "^PYTHON_VERSION="  ${CONF_FILE} | cut -d"=" -f2)
RUNTIME_VERSION=$(shell grep "^RUNTIME_VERSION=" ${CONF_FILE} | cut -d"=" -f2)
FRAMEWORK=$(shell       grep "^FRAMEWORK="       ${CONF_FILE} | cut -d"=" -f2)
BUCKET_NAME=$(shell     grep "^BUCKET_NAME="     ${CONF_FILE} | cut -d"=" -f2)
JOB_FOLDER=$(shell      grep "^JOB_FOLDER="      ${CONF_FILE} | cut -d"=" -f2)
JOB_PREFIX=$(shell      grep "^JOB_PREFIX="      ${CONF_FILE} | cut -d"=" -f2)

# tmp variables
JOB_NAME=${JOB_PREFIX}_$(shell date +'%Y%m%d_%H%M%S')

#   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-.   .-.-
#  / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \ \ / / \
# `-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'   `-`-'
#
# variable extraction tests
#

TATA_PROJECT_ID=$(shell . ./a.sh)
TITI_PROJECT_ID=$(shell tail -n 2 .conf_cat)
TUTU_PROJECT_ID=$(shell grep "^BUCKET_NAME" ${CONF_FILE})
TETE_PROJECT_ID=$(shell grep "^BUCKET_NAME=" ${CONF_FILE} | cut -d"=" -f2)

toto:
	@echo tata ${TATA_PROJECT_ID}
	@echo titi ${TITI_PROJECT_ID}
	@echo tutu ${TUTU_PROJECT_ID}
	@echo tete ${TETE_PROJECT_ID}

# ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^ ~ ^
#
# actual tasks
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
		--job-dir "gs://${BUCKET_NAME}/${JOB_FOLDER}" \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--region ${REGION} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--stream-logs
