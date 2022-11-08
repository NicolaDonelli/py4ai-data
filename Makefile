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
	@echo "  - setup for installing base requirements"
	@echo "  - setup_dev for installing requirements for development"
	@echo "  - format for reformatting files to adhere to PEP8 standards"
	@echo "  - dist for building a tar.gz distribution"
	@echo "  - install for installing the package"
	@echo "  - install_dev for installing the package with development environment"
	@echo "  - reinstall for deleting and reinstalling the package"
	@echo "  - reinstall_dev for deleting and reinstalling the package with development environment"
	@echo "  - uninstall for uninstalling the environment"
	@echo "  - tests for running unittests"
	@echo "  - lint for performing linting"
	@echo "  - mypy for performing static type checking"
	@echo "  - docs for producing documentation in html format"
	@echo "  - checks for running format, mypy, lint and tests altogether"
	@echo "  - clean for removing cache file"
	@echo "  - publish_test for publishing package on TestPyPI"
	@echo "  - publish for publishing package on PyPI"
	@echo "------------------------------------"

$(pre_deps_tag):
	@echo "==Installing pip-tools and black=="
	grep "^pip-tools\|^black"  requirements/requirements_dev.in | xargs ${PYTHON} -m pip install
	grep "^tomli"  requirements/requirements.in | xargs ${PYTHON} -m pip install
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
	${PYTHON} -m mypy --install-types --non-interactive --package py4ai --package tests

tests: setup_dev $(install_tag)
	${PYTHON} -m pytest tests

checks: lint mypy tests
	${PYTHON} -m black --check $(folders)
	${PYTHON} -m isort $(folders) -c

docs: setup_dev $(install_tag) $(doc_files) pyproject.toml
	sphinx-apidoc --implicit-namespaces -f -o sphinx/source/api py4ai
	make --directory=sphinx --file=Makefile clean html

clean:
	@echo "==Cleaning environment=="
	rm -rf docs
	rm -rf build
	rm -rf sphinx/source/api
	rm -rf $(shell find . -name "*.pyc") $(shell find . -name "__pycache__")
	rm -rf *.egg-info .mypy_cache .pytest_cache .make_cache $(env_tag) $(env_dev_tag) $(install_tag)
