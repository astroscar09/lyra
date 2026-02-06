# Lyra

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lyra)](https://pypi.org/project/lyra/)
[![Build](https://img.shields.io/github/actions/workflow/status/astroscar09/lyra/python-package.yml)](https://github.com/astroscar09/lyra/actions)

**Lyra** is a lightweight Python package for performing inference on Lyman-alpha galaxy properties using pre-trained neural density estimators. It allows astronomers and researchers to quickly generate posterior samples and summaries for their datasets with minimal setup.

## Background of Lyra

In astronomy, there is a time in the history of the universe when it underwent a major phase change: the transition from a predominantly neutral universe to a predominantly ionized one. This phase change is called the epoch of reionization (EoR) and is a major topic of study. Knowing the timeline of the EoR can provide clues about the dominant ionizing mechanisms that drove reionization. With our current knowledge, we have a good handle on when reionization ended thanks to the Lyman-alpha forest in quasar spectra; this end is somewhere around a redshift of z ≈ 5.5. However, determining when it started—and its full duration—has been challenging.

Measuring the full duration of the EoR requires estimates of the neutral hydrogen fraction as a function of redshift in order to distinguish between competing reionization models. A common probe used today is the Lyman-alpha emission line due to its hypersensitivity to neutral hydrogen. Lyman-alpha has the unique property that if it encounters a neutral hydrogen atom, it will be absorbed and re-emitted in a random direction. Thus, if we can measure the observed Lyman-alpha emission of galaxies in the EoR and compare that to how much Lyman-alpha they intrinsically emit, we can infer the neutral fraction. However, because of this sensitivity, Lyman-alpha is also affected by internal galaxy dynamics and properties such as dust content and its distribution, making it non-trivial to determine exactly how much Lyman-alpha is being emitted by a galaxy in the EoR.


## Methodology

To circumvent this issue, we take thousands of Lyman-alpha emitting galaxies in the post-reionization universe and map galaxy properties to emergent Lyman-alpha emission. The goal is to directly tie Lyman-alpha emission to galactic observables such as stellar mass, dust, M_UV, UV beta slope, and more. A basic rundown of the methodology is as follows:

- Cross-match multiple photometric catalogs to a spectroscopic catalog with millions of entries
- Run a photometric redshift estimation code to determine the likelihood of each matched source being between redshifts 1.9–3.5; at this step we obtain the redshift probability distribution P(z)
- Use external information such as other Lyman-alpha likelihood estimates and combine it with the P(z) information to uniquely determine whether a galaxy is a Lyman-alpha emitter
- Use an SED-fitting code to determine the galaxy properties
- Lastly, fold the galaxy properties into a normalizing flow framework to map galaxy properties to emergent Lyman-alpha properties

## Skills Used

- Automated ingestion, processing, and cleaning of data
- Parallel computing on a supercomputing cluster
- Bayesian analysis and techniques to identify Lyman-alpha emitters
- Machine Learning validation, inference, and diagnostics using Python
- Simulation-based inference using PyTorch and normalizing flows on the backend
- Training conducted at scale using a supercomputing cluster


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


