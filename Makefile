# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash, keep in mind that in the Makefile there's top-level Makefile-only syntax, and everything else is bash script syntax.
PYTHON = python

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help

folders := py4ai/data tests
files := $(shell find . -name "*.py")
doc_files := $(shell find sphinx -name "*.*")

# Uncomment to store cache installation in the environment
# package_dir := $(shell python -c 'import site; print(site.getsitepackages()[0])')
package_dir := .make_cache
package_name=$(shell python -c "import tomli;from pathlib import Path;print(tomli.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['name'])")

$(shell mkdir -p $(package_dir))

pre_deps_tag := $(package_dir)/.pre_deps
env_tag := $(package_dir)/.env_tag
env_dev_tag := $(package_dir)/.env_dev_tag
install_tag := $(package_dir)/.install_tag

# ======================
# Rules and Dependencies
# ======================

help:
	@echo "---------------HELP-----------------"
	@echo "Package Name: $(package_name)"
	@echo " "
	@echo "Type 'make' followed by one of these keywords:"
	@echo " "
	@echo "  - reqs_dev to build closed development requirements, requirements/requirements_dev.txt, from requirements/requirements_dev.in and requirements/requirements.in"
	@echo "  - reqs to build closed minimal requirements, requirements/requirements.txt, from requirements/requirements.in"
	@echo "  - setup to install minimal requirements"
	@echo "  - setup_dev to install development requirements"
	@echo "  - format to reformat files to adhere to PEP8 standards"
	@echo "  - dist to build a tar.gz distribution"
	@echo "  - install to install the package with minimal requirements"
	@echo "  - install_dev to install the package with development environment"
	@echo "  - uninstall to uninstall the package and its dependencies"
	@echo "  - tests to run unittests using pytest as configured in pyproject.toml"
	@echo "  - lint to perform linting using flake8 as configured in pyproject.toml"
	@echo "  - mypy to perform static type checking using mypy as configured in pyproject.toml"
	@echo "  - bandit to find security issues in app code using bandit as configured in pyproject.toml"
	@echo "  - licensecheck to check dependencies licences compatibility with application license using licensecheck as configured in pyproject.toml"
	@echo "  - docs to produce documentation in html format using sphinx as configured in pyproject.toml"
	@echo "  - checks to run mypy, lint, bandit, licensecheck, tests and check formatting altogether"
	@echo "  - clean to remove cache file"
	@echo "------------------------------------"

$(pre_deps_tag):
	@echo "==Installing pip-tools and black=="
	${PYTHON} -m pip install --upgrade --quiet pip
	grep "^pip-tools\|^black"  requirements/requirements_dev.in | xargs ${PYTHON} -m pip install --quiet
	grep "^tomli\|^setuptools"  requirements/requirements.in | xargs ${PYTHON} -m pip install --upgrade --quiet
	touch $(pre_deps_tag)

requirements/requirements.txt: requirements/requirements_dev.txt
	@echo "==Compiling requirements.txt=="
	cat requirements/requirements.in > subset.in
	echo "" >> subset.in
	echo "-c requirements/requirements_dev.txt" >> subset.in
	pip-compile --output-file "requirements/requirements.txt" --quiet --no-emit-index-url subset.in
	rm subset.in

reqs: requirements/requirements.txt

requirements/requirements_dev.txt: requirements/requirements_dev.in requirements/requirements.in $(pre_deps_tag)
	@echo "==Compiling requirements_dev.txt=="
	pip-compile --output-file requirements/requirements_dev.txt --quiet --no-emit-index-url requirements/requirements.in requirements/requirements_dev.in

reqs_dev: requirements/requirements_dev.txt

$(env_tag): requirements/requirements.txt
	@echo "==Installing requirements.txt=="
	pip-sync --quiet requirements/requirements.txt
	rm -rf $(install_tag)
	touch $(env_tag)

$(env_dev_tag): requirements/requirements_dev.txt
	@echo "==Installing requirements_dev.txt=="
	pip-sync --quiet requirements/requirements_dev.txt
	rm -rf $(install_tag)
	touch $(env_dev_tag)

setup: $(env_tag)
	@echo "==Setting up package environment=="
	rm -f $(env_dev_tag)

setup_dev: $(env_dev_tag)
	@echo "==Setting up development environment=="
	rm -f $(env_tag)

dist/.build-tag: $(files) pyproject.toml requirements/requirements.txt
	@echo "==Building package distribution=="
	${PYTHON} setup.py --quiet sdist
	ls -rt  dist/* | tail -1 > dist/.build-tag

dist: dist/.build-tag setup.py

$(install_tag): dist/.build-tag
	@echo "==Installing package=="
	${PYTHON} -m pip install --quiet --no-deps $(shell ls -rt  dist/*.tar.gz | tail -1)
	touch $(install_tag)

uninstall:
	@echo "==Uninstall package $(package_name)=="
	pip uninstall -y $(package_name)
	pip freeze | grep -v "@" | xargs pip uninstall -y
	rm -f $(env_tag) $(env_dev_tag) $(pre_deps_tag) $(install_tag)

install: $(install_tag)

format: setup_dev
	${PYTHON} -m black $(folders)
	${PYTHON} -m isort $(folders)

lint: setup_dev
	${PYTHON} -m flake8 $(folders)

mypy: setup_dev $(install_tag)
	${PYTHON} -m mypy --install-types --non-interactive --package tests --package py4ai

tests: setup_dev $(install_tag)
	${PYTHON} -m pytest tests

bandit: setup_dev
	${PYTHON} -m bandit -r -c pyproject.toml --severity-level high --confidence-level high .

licensecheck: setup_dev
	${PYTHON} -m licensecheck --zero

checks: lint mypy bandit licensecheck tests
	${PYTHON} -m black --check $(folders)
	${PYTHON} -m isort $(folders) -c

docs: setup_dev $(install_tag) $(doc_files) pyproject.toml
	sphinx-apidoc --implicit-namespaces -f -d 20 -M -e -o sphinx/source/api py4ai
	make --directory=sphinx --file=Makefile clean html

clean:
	@echo "==Cleaning environment=="
	rm -rf docs
	rm -rf build
	rm -rf sphinx/source/api
	rm -rf $(shell find . -name "*.pyc") $(shell find . -name "__pycache__")
	rm -rf *.egg-info .mypy_cache .pytest_cache .make_cache $(env_tag) $(env_dev_tag) $(install_tag)
