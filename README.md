# Lyra

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lyra)](https://pypi.org/project/lyra/)
[![Build](https://img.shields.io/github/actions/workflow/status/astroscar09/lyra/python-package.yml)](https://github.com/astroscar09/lyra/actions)

**Lyra** is a lightweight Python package for performing inference on Lyman-alpha galaxy properties using pre-trained neural density estimators. It allows astronomers and researchers to quickly generate posterior samples and summaries for their datasets with minimal setup.

---

## Background of Lyra

In astronomy, there is a time in the history of the universe when it underwent a major phase change: the transition from a predominantly neutral universe to a predominantly ionized one. This phase change is called the epoch of reionization (EoR) and is a major topic of study. Knowing the timeline of the EoR can provide clues about the dominant ionizing mechanisms that drove reionization. With our current knowledge, we have a good handle on when reionization ended thanks to the Lyman-alpha forest in quasar spectra; this end is somewhere around a redshift of z ≈ 5.5. However, determining when it started—and its full duration—has been challenging.

Measuring the full duration of the EoR requires estimates of the neutral hydrogen fraction as a function of redshift in order to distinguish between competing reionization models. A common probe used today is the Lyman-alpha emission line due to its hypersensitivity to neutral hydrogen. Lyman-alpha has the unique property that if it encounters a neutral hydrogen atom, it will be absorbed and re-emitted in a random direction. Thus, if we can measure the observed Lyman-alpha emission of galaxies in the EoR and compare that to how much Lyman-alpha they intrinsically emit, we can infer the neutral fraction. However, because of this sensitivity, Lyman-alpha is also affected by internal galaxy dynamics and properties such as dust content and its distribution, making it non-trivial to determine exactly how much Lyman-alpha is being emitted by a galaxy in the EoR.

---

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

- **Ensemble inference** — each model key loads 5 independently trained neural density estimators; samples are pooled across all 5 and a KDE is used to find the posterior mode alongside 16th/50th/84th percentile credible intervals.
- **14 pre-trained ensemble models** covering feature sets ranging from 2 inputs (M_UV, β) up to 10 inputs, available in both MAF and NSF flow architectures.
- **Train your own model** — bring your own LAE data, train a new SBI model with one function call, and it is automatically registered so you can load it immediately with `load_model()`.
- **Dataset downloads** — download the LAE training set or the full LAE ML catalog on demand; the package ships without large data files.
- **Lazy model downloading** — only the 5 pkl files for the model you request are downloaded and cached; the other ensembles are never touched until you ask for them.
- **Fully pip-installable** with HuggingFace-backed model and dataset storage.

---

## Installation

### Recommended Setup

This approach uses conda to handle complex dependencies (PyTorch, SBI), then registers the package locally:

```bash
conda env create -f environment.yml
conda activate lyra_env
pip install -e .
```

**Important:**
- Always activate `lyra_env` before using Lyra.
- `pip install -e .` must be run while the conda environment is active.
- To control where HuggingFace caches downloaded models, set the `HF_HOME` environment variable:

```bash
export HF_HOME=/Users/username/Desktop/lyra/lyra/models
```

### Troubleshooting

1. Ensure the conda environment is active: `conda activate lyra_env`
2. Reinstall if needed: `pip install -e . --force-reinstall`
3. Verify dependencies: `conda list` (should show torch, sbi, etc.)

---

## Usage

### Ensemble Inference (default)

`Lyra()` with no arguments enters ensemble mode. Call `load_model()` with one of the available keys to download and cache the 5-model ensemble for that configuration. The first call downloads the files from HuggingFace; subsequent calls load from the local cache instantly.

```python
import numpy as np
from lyra import Lyra

lyra = Lyra()               # prints all available model keys
lyra.load_model('Muv_beta_zuko_maf')

# data shape: (N_galaxies, N_features) — must match the model's feature set
data = np.array([[-20.5, -1.8],
                 [-19.0, -2.1],
                 [-21.2, -1.3]])

samples, summary = lyra.sample(data, num_samples=10000)

print(summary)
#    max_ew   p16_ew   p50_ew   p84_ew
# 0    2.31     1.98     2.28     2.51
# 1    2.65     2.33     2.56     2.77
# 2    1.80     1.57     1.84     2.12
```

`summary` contains four columns per galaxy:
- `max_ew` — posterior mode (peak of the KDE across all 5 models)
- `p16_ew`, `p50_ew`, `p84_ew` — 16th, 50th, and 84th percentile credible intervals

You can request a different percentile set:

```python
samples, summary = lyra.sample(data, percentiles=(2.5, 16, 50, 84, 97.5))
```

### Available Ensemble Models

| Model key | Input features | Flow |
|---|---|---|
| `Muv_beta_zuko_maf` | M_UV, β | MAF |
| `Muv_beta_zuko_nsf` | M_UV, β | NSF |
| `Muv_stellar_mass_beta_ssfr_zuko_maf` | M_UV, M★, β, sSFR | MAF |
| `Muv_stellar_mass_beta_ssfr_zuko_nsf` | M_UV, M★, β, sSFR | NSF |
| `beta_ssfr_Muv_burstiness_zuko_maf` | β, sSFR, M_UV, burstiness | MAF |
| `beta_ssfr_Muv_burstiness_zuko_nsf` | β, sSFR, M_UV, burstiness | NSF |
| `log_Av_logU_ssfr_stellar_mass_beta_Muv_burstiness_zuko_maf` | log Av, log U, sSFR, M★, β, M_UV, burstiness | MAF |
| `log_Av_logU_ssfr_stellar_mass_beta_Muv_burstiness_zuko_nsf` | log Av, log U, sSFR, M★, β, M_UV, burstiness | NSF |
| `log_Av_ssfr_Muv_burstiness_zuko_maf` | log Av, sSFR, M_UV, burstiness | MAF |
| `log_Av_ssfr_Muv_burstiness_zuko_nsf` | log Av, sSFR, M_UV, burstiness | NSF |
| `log_Z_metal_beta_logU_stellar_mass_ssfr_Muv_burstiness_zuko_maf` | log Z, β, log U, M★, sSFR, M_UV, burstiness | MAF |
| `log_Z_metal_beta_logU_stellar_mass_ssfr_Muv_burstiness_zuko_nsf` | log Z, β, log U, M★, sSFR, M_UV, burstiness | NSF |
| `log_Z_metal_log_Av_B_delta_logU_stellar_mass_ssfr_beta_Muv_burstiness_zuko_maf` | log Z, log Av, B, δ, log U, M★, sSFR, β, M_UV, burstiness | MAF |
| `log_Z_metal_log_Av_B_delta_logU_stellar_mass_ssfr_beta_Muv_burstiness_zuko_nsf` | log Z, log Av, B, δ, log U, M★, sSFR, β, M_UV, burstiness | NSF |

### Single-Model Mode (for retrained or custom models)

If you have trained your own model (see Training below), you can load it directly by passing the path to the pkl file:

```python
lyra = Lyra('path/to/my_model.pkl')
lyra.sample(data, num_samples=1000)

print(lyra.summary_df)   # percentile summary
print(lyra.post_samples) # raw posterior samples
```

---

### Training a New Model

Lyra includes a training module so you can retrain on new data or extend an existing ensemble with your own observations. The trained model is automatically registered in your local config so it is immediately accessible via `load_model()`.

```python
import pandas as pd
from lyra import train_model, Lyra

# Your data as a DataFrame with feature columns and a log10(EW) target column
df = pd.read_csv('my_lae_catalog.csv')

path = train_model(
    data           = df,
    feature_cols   = ['Muv', 'beta'],
    label_col      = 'log_ew',
    model_name     = 'my_ensemble',
    density_estimator = 'zuko_maf',   # or 'zuko_nsf'
    gauss          = True,            # recommended: apply Gaussian CDF transform to target
    ew_bounds      = (0, 3.4),        # physical range of log10(EW) in your data
    output_dir     = './my_models',
)
# Model saved to: ./my_models/my_ensemble_zuko_maf.pkl
# Registered 'my_ensemble' — access it with: lyra.load_model('my_ensemble')
```

Load it immediately for inference:

```python
lyra = Lyra()
lyra.load_model('my_ensemble')
samples, summary = lyra.sample(data)
```

**Key training parameters:**

| Parameter | Default | Description |
|---|---|---|
| `density_estimator` | `'zuko_maf'` | Flow architecture: `'zuko_maf'` or `'zuko_nsf'` |
| `gauss` | `True` | Apply Gaussian CDF transform to target before training |
| `ew_bounds` | `(0, 3.4)` | Physical EW range (log10 units) — used as CDF transform bounds |
| `gauss_prior_bounds` | `(-5, 5)` | Prior range in Gaussian-transformed space |
| `hidden_features` | `50` | Number of hidden features in the flow network |
| `num_transforms` | `5` | Number of flow transform layers |
| `test_size` | `None` | Fraction of data to hold out for validation |

> **Note:** The `gauss=True` setting (recommended) applies an inverse Gaussian CDF transform to the log EW values before training, mapping them to an approximately unbounded Gaussian space. This improves flow training stability. At inference time, Lyra automatically applies the forward CDF transform to map samples back to physical EW space. If you set `gauss=False`, raw EW values are used directly and no back-transform is applied.

---

### Downloading the LAE Datasets

The package ships without data files. Two datasets are available for download on demand.

#### Training Set

```python
from lyra import download_training_data

path = download_training_data(dest_dir='./data')
```

This is the subset of the full LAE ML catalog used to train the Lyra SBI models. It was constructed to be as **uniform as possible in log₁₀ Lyman-alpha equivalent width** so that no EW range has disproportionate influence on training. It contains ~449,000 galaxies and 37 columns of galaxy properties.

#### Full LAE ML Dataset

```python
from lyra import download_full_LAE_ML_data

path = download_full_LAE_ML_data(dest_dir='./data')
```

This is the complete LAE ML catalog (~1.8 million galaxies, 32 columns) from which the training set was drawn. It is not EW-uniform — the EW distribution reflects the natural distribution of Lyman-alpha emitters in the underlying photometric and spectroscopic surveys. This dataset is useful if you want to augment the training set with additional sources or study the full population distribution.

> **Relationship between the two datasets:** The training set is a uniform-EW subsample of the full catalog. If you want to retrain Lyra with new data, you may find it helpful to merge your sources with the training set (not the full catalog) to preserve EW uniformity during training.

Both functions will show the file size and ask for confirmation before downloading. Downloads are cached by HuggingFace Hub, so re-running the function does not re-download the file.

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Citation

If you use Lyra in your research, please cite:

Oscar A. Chavez Ortiz, Lyra: A Python package for Lyman-alpha Galaxy Inference, 2026
