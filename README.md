# Python Package Template

[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/release.yaml)
[![github](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml/badge.svg)](https://github.com/projectronin/ronin-blueprint-python-lib/actions/workflows/main.yaml)
[![codecov](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib/branch/main/graph/badge.svg?token=z6l3Vet7N6)](https://codecov.io/gh/projectronin/ronin-blueprint-python-lib)

Use this template to bootstrap a Python library at Project Ronin.

---

## Getting Started Guide

### Step 1: Create a Repository using this Template

To begin, create a new repository using this template
following [this guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
from GitHub.

Please use the following naming convention when naming the new repository on GitHub.

`{TEAM}-lib-{NAME}`

- `{TEAM}` signals the owner of this repository.
  [This article](https://projectronin.atlassian.net/wiki/spaces/ENG/pages/1649410056/GitHub+Guidelines) provides
  additional information on team abbreviation conventions.
- `lib` signals that this is a library distributed as a package.
- `{NAME}` should provide relevant context to the intent of this project.

For example:

- `infx-lib-awesome-client` (Informatics / Library / Awesome Client)
- `ds-lib-super-duper-model` (Data Science / Library / Super Duper Model)
- `dp-lib-mind-blowing-util` (Data Platform / Library / Mind Blowing Utility)

### Step 2: Update the Package Name

Next, update all references to `ds-transport-calibration` and `ds_transport_calibration` with your package name (`{NAME}`) using the established
casing conventions.

For example:

- `awesome-client`, `awesome_client`
- `super-duper-model`, `super_duper_model`
- `mind-blowing-util`, `mind_blowing_util`

### Step 3: Set Branch Protection Rules

Follow [this guide](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/managing-a-branch-protection-rule#creating-a-branch-protection-rule)
to go to the settings for branch protection rules.

Click "Add rule."

Set the "Branch name pattern" to *main*.

Enable "Require a pull request before merging."

Enable "Require approvals."

Set "Required number of approvals before merging" to *1*.

Enable "Require status checks to pass before merging."

### Step 4: Set up [codecov.io](codecov.io)

Retrieve the repository upload token from [codecov.io](codecov.io) for your new repository
using [this guide](https://docs.codecov.com/docs/codecov-uploader#upload-token).

Then, create a new action secret for your new repository token named `CODECOV_TOKEN`
using [this guide](https://docs.github.com/en/actions/security-guides/encrypted-secrets#creating-encrypted-secrets-for-a-repository)
. Set the value to the repository upload token from [codecov.io](codecov.io).

Test this setting by triggering the `main` GitHub Action for your new repository
using [this guide](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow). Next, check
the `coverage` step to verify that the coverage upload to [codecov.io](codecov.io) is successful.

### Step 5: Cleanup

Once complete, please delete this section ([Getting Started Guide](#getting-started-guide)) of the README.md from your
new repository. Then reference [CONTRIBUTING.md](CONTRIBUTING.md) for next steps.

---

## Installation

```bash
pip install ds-transport-calibration
```

## Usage

```python
from ds_transport_calibration import say_hello

msg = say_hello("World")

print(msg)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on contributing to this project.
