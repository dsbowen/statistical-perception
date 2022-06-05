PORT_TEST ?= 8080
REPORTS_DIR ?= reports
SRC_DIR ?= src

# Format the src directory with black
.PHONY: format
format:
	black ${SRC_DIR}
	black tests

# Run unit tests, produce a coverage report, and serve it as a web page
.PHONY: test
test:
	mkdir -p ${REPORTS_DIR}
	tox
	coverage html -d ${REPORTS_DIR}/coverage
	coverage report
.PHONY: testserve
testserve:
	python -m http.server ${PORT_TEST} -d ${REPORTS_DIR}/coverage

# Lint the source directory with pylint
.PHONY: lint
lint:
	mkdir -p ${REPORTS_DIR}
	pylint ${SRC_DIR} --reports=y > ${REPORTS_DIR}/lint.txt

# Run typehint checks with mypy
.PHONY: typehint
typehint:
	mypy ${SRC_DIR}

# Run formatting, unit tests, linting, and typehints
.PHONY: pipeline
pipeline: format test lint typehint
