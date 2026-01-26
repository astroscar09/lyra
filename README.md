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
- Fully pip-installable and uses HuggingFace to store and download the SBI model.

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

**Important:** 

- Always activate the `lyra_env` conda environment before using Lyra. 
- The `pip install -e .` command must be run while the conda environment is active to ensure all dependencies are properly linked.
- The package uses huggingface_hub to download the models to mitigate large files on downloading the package, please set up an environment variable HF_HOME point to the lyra/models/ directory. 

```bash
Ex: export HF_HOME=/Users/username/Desktop/lyra/lyra/models
```

Available Models Ready for Download from HuggingFace: 

- full_SBI_NPE_Muv_beta.pkl
- full_SBI_NPE_beta_ssfr_Muv_burst.pkl
- full_SBI_NPE_Muv_mass_beta_ssfr.pkl
- full_SBI_NPE_Av_logU_ssfr_mass_beta_Muv_burst.pkl
- full_SBI_NPE_metallicity_beta_logU_mass_ssfr_Muv_burst.pkl
- full_SBI_NPE_metallicity_Av_B_delta_logU_mass_ssfr_beta_Muv_burst.pkl

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


