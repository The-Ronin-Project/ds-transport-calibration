# Install project dependencies.
#
# Warning: Before executing the install target the Conda environment must be created and activated. See CONTRIBUTING.md
# for additional instructions.
#
# The following actions are performed:
# 	1. Update the Conda environment as defined in ./environment.yaml
# 	2. Install or upgrade all project dependencies as defined in ./setup.cfg.
#	3. Install or upgrade pre-commit
#	4. Execute pre-commit install, setting up the pre-commit hooks.
#
.PHONY: install
install:
	@conda env update -f environment.yaml
	@python -m pip install -e ".[test]"
	@python -m pip install --upgrade pre-commit
	@pre-commit install

# Lint using Flake8.
#
# The Flake8 styleguide is defined in ./.flake8.
#
.PHONY: lint
lint:
	@python -m pip install --upgrade flake8
	@flake8 src tests

# Format using Black.
#
# The default Black styleguide is used.
#
.PHONY: format
format:
	@python -m pip install --upgrade black
	@python -m black .

# Unit test using the default Python unit testing framework, unittest, while collecting coverage using Coverage.py.
#
# The default unit testing configuration is used. Coverage.py configuration is defined in ./.coveragerc.
#
# The following actions are performed:
#	1. Install or upgrade Coverage.py
#	2. Execute unit testing and collect coverage
#	3. Create a Coverage.py report.
#	4. Create a Coverage.py report in XML format.
#
.PHONY: test
test:
	@python -m pip install --upgrade coverage
	@coverage run -m unittest
	@coverage report
	@coverage xml

# Create a distribution according to PEP-517 guidelines using build.
#
# The build configuration is defined in ./pyproject.toml. The resulting package is located at ./dist.
#
.PHONY: package
package:
	@python -m pip install --upgrade build
	@rm -rf dist src/*.egg-info
	@python -m build

# Validate the distribution.
#
# A simple health check is executed to lightly validate that the package was built correctly.
#
# The following actions are preformed:
#	1. Install the distribution from the Wheel package.
#	2. Execute a simple python program that imports the top-level package.
#	3. Uninstall the package.
#
.PHONY: validate
validate:
	@python -m pip install --force-reinstall ./dist/*.whl
	@python -c "import transport_calibration"
	@python -m pip uninstall -y ds-transport-calibration

# Release the package using Twine.
#
# Note: This command is intended to be executed from the release GitHub action, defined at
# ./github/workflows/release.yaml. If a different release mechanism is required, this target must be supplied with Twine
# credentials.
#
.PHONY: release
release:
	@python -m pip install --upgrade twine
	@twine upload dist/*

# Clean the repository.
#
# Deletes all transient project files, resulting in clean project state.
#
# The following files are deleted:
# 	- ./.coverage - The Coverage.py report.
# 	- ./.dist - The build distribution.
# 	- ./src/*.egg_info - Project metadata created by build.
.PHONY: clean
clean:
	@rm -rf .coverage coverage.xml dist src/*.egg-info
