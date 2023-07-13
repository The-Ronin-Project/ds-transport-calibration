## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing
purposes. See deployment for notes on deploying the project on a live system.

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing](#installing)
3. [Development](#development)
4. [Testing](#testing)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Deployment](#deployment)
7. [Template Updates](#template-updates)
8. [Code Coverage](#code-coverage)

## Prerequisites

Requirements for the software and other tools to build, test, and push.

- [Conda](https://docs.conda.io/en/latest/)

## Installing

Here is a step-by-step guide to getting up and running

Create a Conda virtual environment using the following `Makefile` target.

    conda env create -f environment.yaml

Activate the virtual environment.

    conda activate hello-world

Finally, execute the following `Makefile` target to install all dependencies required for project development.

    make install

## Development

### Writing source code

All project development occurs within the established Python module located at [./src/hello_world](./src/hello_world).
When developing source code, please use relative imports to reference other source code. This ensures source code is
referencing source code and not an installed package with the same name.

For example:

```python
from . import say_hello

say_hello("World")
```

### Top-level code environment

Top-level level code can be defined using `__main__`. For example, when developing a command-line-interface. Note, you
cannot run this script from PyCharm using the run command. Instead, execute the following process to invoke `__main__`.

    make package
    python -m pip install --force-reinstall ./dist/*.whl
    python -m hello_world

## Testing

### Writing tests

Tests are written in the [./tests](./tests) directory using the `unittest` framework.

When writing tests, please use absolute imports to the [src](./src) directory (
i.e. `from src import ...`). This ensures that tests are executed against the source code and not an installed package
with the same name. Please refer to [PEP 328](https://www.python.org/dev/peps/pep-0328/) to rational between relative
and absolute imports.

For example:

```python
import unittest

from src.hello_world import say_hello


class TestSayHello(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(say_hello("World"), "Hello, World!")
```

### Running the tests

To simplify invocation, use the following `Makefile` target. This will execute all available tests and report the total
test coverage.

    make test

## Submitting a Pull Request

Please submit a pull request using the provided template on GitHub.

## Deployment

Deployments are handled by GitHub Actions on Git tagging using [Semantic Versioning](https://semver.org/). Read more
about tagging [here](https://git-scm.com/book/en/v2/Git-Basics-Tagging).

There are two options to trigger a release.

1. Manually create a release on GitHub. This is the preferred option.
2. Manually create a tag and push.

When creating a release on GitHub, create a new tag via the dropdown. Do not prefix the tag with a 'v'. Python does not
follow this convention, and doing so will not trigger a release. Use the "auto-generate release notes" button to create
release notes based. To create a release on GitHub please
use [this guide](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
.

To manually create a tag use the following steps.

    git tag <MAJOR>.<MINOR>.<PATCH>
    git push origin <MAJOR>.<MINOR>.<PATCH>

If you need to build the package manually for testing, use the following `Makefile` target.

    make package

## Code Coverage

Your pull request must meet code coverage requirements to merge as defined in [./codecov.yaml](./codecov.yaml). In
summary:

- Total coverage cannot drop below 80%
- The change in coverage cannot differ by more than 1% from the main branch

While it is not required, please strive for 100% coverage. If there is a practical reason to not check coverage for
sections of code, please use [this guide](https://coverage.readthedocs.io/en/6.3.2/excluding.html) to exclude the code
from [coverage.py](https://coverage.readthedocs.io/).

## Template Updates

After forking this repository, it's possible to get updates from the template by using `git merge`. The following will
allow you to merge updates from the template repository.

    git remote add template git@github.com:projectronin/ronin-blueprint-python-lib.git
    git fetch template main
    git checkout -b get-updates-from-template
    git merge --allow-unrelated-histories --no-commit --squash template/main
    git commit
    git push -u origin merge-updates-from-template

