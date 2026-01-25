# Lyra

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lyra)](https://pypi.org/project/lyra/)
[![Build](https://img.shields.io/github/actions/workflow/status/astroscar09/lyra/python-package.yml)](https://github.com/astroscar09/lyra/actions)

**Lyra** is a lightweight Python package for performing inference on Lyman-alpha galaxy properties using pre-trained neural density estimators. It allows astronomers and researchers to quickly generate posterior samples and summaries for their datasets with minimal setup.

---

## Features

- Load and run pre-trained models for Lyman-alpha galaxy property inference.
- Supports multiple models via simple file input.
- Generates posterior samples and summary statistics.
- Includes plotting utilities for visualizing input vs output with uncertainties.
- Fully pip-installable, no SQL or external database dependencies.

---

## Installation

### Recommended Setup (Safe & Works From Anywhere)
This approach uses conda to handle complex dependencies, then registers your package locally:

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate lyra_env

# Install Lyra in editable mode (allows running from any directory)
# Must run this AFTER activating the conda environment
pip install -e .
```

**Important:** Always activate the `lyra_env` conda environment before using Lyra. The `pip install -e .` command must be run while the conda environment is active to ensure all dependencies are properly linked.

### Troubleshooting
If you encounter issues:
1. Ensure the conda environment is active: `conda activate lyra_env`
2. Reinstall if needed: `pip install -e . --force-reinstall`
3. Verify dependencies: `conda list` (should show torch, sbi, etc.)



## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use Lyra in your research, please cite:

Oscar A. Chavez Ortiz, Lyra: A Python package for Lyman-alpha Galaxy Inference, 2026


