import io
import pickle
import yaml
from pathlib import Path

import numpy as np
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sklearn.model_selection import train_test_split

from lyra.evaluation import inv_cdf_transform

_USER_CONFIG_PATH = Path.home() / ".lyra" / "user_models.yaml"


def train_model(
    data,
    feature_cols,
    label_col,
    model_name,
    density_estimator="zuko_maf",
    output_dir=".",
    gauss=True,
    ew_bounds=(0, 3.4),
    gauss_prior_bounds=(-5, 5),
    hidden_features=50,
    num_transforms=5,
    num_bins=8,
    test_size=None,
    random_state=423,
) -> Path:
    """Train a single SBI model and register it for use with Lyra.

    Args:
        data: pandas DataFrame containing features and label columns.
        feature_cols: List of column names to use as inputs.
        label_col: Column name for the target (log EW).
        model_name: Base name for the saved model and its ensemble entry.
        density_estimator: SBI flow type — 'zuko_maf' or 'zuko_nsf'.
        output_dir: Directory to save the trained pkl file.
        gauss: If True, apply inv_cdf_transform to the target before training
               and use a Gaussian prior in transformed space. If False, use
               raw target values with a uniform prior over ew_bounds.
        ew_bounds: Physical EW range (log units). Used as prior bounds when
                   gauss=False, and as the CDF transform range when gauss=True.
        gauss_prior_bounds: Prior bounds in Gaussian-transformed space.
                            Only used when gauss=True.
        hidden_features: Number of hidden features in the flow network.
        num_transforms: Number of transform layers.
        num_bins: Number of bins for NSF flows.
        test_size: If given, hold out this fraction of data for validation.
                   If None, all data is used for training.
        random_state: Random seed for reproducibility.

    Returns:
        Path to the saved pkl file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = data.copy()

    if gauss:
        df["_ew_transformed"] = inv_cdf_transform(df[label_col].values, ew_bounds)
    else:
        df["_ew_transformed"] = df[label_col].values

    if test_size is not None:
        train_df, _ = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        train_df = df

    x = _to_tensor(train_df[feature_cols].values)
    theta = _to_tensor(train_df["_ew_transformed"].values.reshape(-1, 1))

    prior = _define_prior(gauss, ew_bounds, gauss_prior_bounds)
    inference = _set_up_sbi(prior, density_estimator, hidden_features, num_transforms, num_bins)
    _train_sbi(inference, theta, x)

    fname = f"{model_name}_{density_estimator}.pkl"
    save_path = output_dir / fname
    _save_model(inference, save_path)
    print(f"Model saved to: {save_path}")

    register_model(
        save_path,
        ensemble_name=model_name,
        gauss=gauss,
        ew_bounds=ew_bounds,
        gauss_prior_bounds=gauss_prior_bounds,
    )

    return save_path


def register_model(
    model_path,
    ensemble_name,
    gauss=True,
    ew_bounds=(0, 3.4),
    gauss_prior_bounds=(-5, 5),
    config_path=None,
) -> None:
    """Register a trained model pkl into the user model config.

    Creates or updates an ensemble entry in the user config YAML so the
    model can be loaded via lyra.load_model(ensemble_name).

    Args:
        model_path: Path to the saved pkl file.
        ensemble_name: Key name for this ensemble in the config.
        gauss: Whether this model uses the Gaussian CDF transform.
        ew_bounds: Physical EW range used during training.
        gauss_prior_bounds: Gaussian prior bounds used during training.
        config_path: Path to user config YAML. Defaults to ~/.lyra/user_models.yaml.
    """
    config_path = Path(config_path) if config_path else _USER_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if config_path.exists():
        with open(config_path) as f:
            existing = yaml.safe_load(f) or {}

    model_path = Path(model_path).resolve()

    if ensemble_name in existing:
        existing[ensemble_name]["files"].append(str(model_path))
    else:
        existing[ensemble_name] = {
            "gauss_transform": gauss,
            "ew_bounds": list(ew_bounds),
            "gauss_prior_bounds": list(gauss_prior_bounds),
            "files": [str(model_path)],
        }

    with open(config_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    print(f"Registered '{ensemble_name}' in {config_path}")
    print(f"Access it with: lyra = Lyra(); lyra.load_model('{ensemble_name}')")


def _define_prior(gauss, ew_bounds, gauss_prior_bounds):
    if gauss:
        return BoxUniform(
            low=torch.tensor([gauss_prior_bounds[0]], dtype=torch.float32),
            high=torch.tensor([gauss_prior_bounds[1]], dtype=torch.float32),
        )
    return BoxUniform(
        low=torch.tensor([ew_bounds[0]], dtype=torch.float32),
        high=torch.tensor([ew_bounds[1]], dtype=torch.float32),
    )


def _set_up_sbi(prior, density_estimator, hidden_features, num_transforms, num_bins):
    return NPE(density_estimator=density_estimator, prior=prior)


def _train_sbi(inference, theta, x):
    inference.append_simulations(theta, x).train()
    return inference


def _save_model(inference, path):
    with open(path, "wb") as f:
        pickle.dump(inference, f)


def _to_tensor(data):
    return torch.tensor(np.asarray(data, dtype=np.float32))
