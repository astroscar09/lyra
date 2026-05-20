from lyra import config
import torch
import numpy as np
from tqdm import tqdm
from lyra.io_utils import grab_model_file, load_model, fetch_ensemble_models, load_posterior_from_path
import pandas as pd
from lyra.evaluation import (
    sample_from_posterior,
    grab_percentiles_from_posterior,
    cdf_transform,
    make_kde,
)
from pathlib import Path
import yaml


_USER_CONFIG_PATH = Path.home() / ".lyra" / "user_models.yaml"


class Lyra():
    """
    Bayesian inference for Lyman-alpha galaxy properties using pre-trained SBI models.

    Two usage modes:

    Ensemble mode (default):
        lyra = Lyra()
        lyra.load_model('Muv_beta_zuko_maf')
        samples, df = lyra.sample(data)

    Single-model mode (for custom / retrained models):
        lyra = Lyra('path/to/my_model.pkl')
        lyra.sample(data, num_samples=1000)
    """

    def __init__(self, model_file=None):
        if model_file is None:
            self._mode = "ensemble"
            self.posteriors = None
            self.model_key = None
            self._gauss_transform = None
            self._ew_bounds = None
            self._gauss_prior_bounds = None
            self.summary_df = None
            self.post_samples = None
            self._ensemble_config = None
            self._load_ensemble_config()
        else:
            self._mode = "single"
            self.posterior = None
            self.summary_df = None
            self.data = None
            self.post_samples = None
            self.load_model_schema()
            self.model_key = grab_model_file(model_file)
            self.grab_model(model_file)
            self.check_model_is_loaded()
            self.init_display_message(self.model_key)

    def __repr__(self):
        return f"<Lyra mode={self._mode} model_key={self.model_key}>"

    # ------------------------------------------------------------------
    # Ensemble mode methods
    # ------------------------------------------------------------------

    def _load_ensemble_config(self):
        model_dir = Path(__file__).parent / "models"
        package_config = model_dir / "model_config_LAE.yaml"
        with open(package_config) as f:
            self._ensemble_config = yaml.safe_load(f)

        if _USER_CONFIG_PATH.exists():
            with open(_USER_CONFIG_PATH) as f:
                user_config = yaml.safe_load(f) or {}
            self._ensemble_config.update(user_config)

        print("Available ensemble models:")
        for key in self._ensemble_config:
            print(f"  {key}")
        print('\nLoad a model with: lyra.load_model("<model_key>")')

    def load_model(self, model_key: str):
        """Load an ensemble by key name. Only available in ensemble mode."""
        if self._mode != "ensemble":
            raise RuntimeError(
                "load_model() is only available in ensemble mode. "
                "Instantiate Lyra without arguments: Lyra()"
            )

        if model_key not in self._ensemble_config:
            available = list(self._ensemble_config.keys())
            raise KeyError(
                f"Model '{model_key}' not found. Available models:\n"
                + "\n".join(f"  {k}" for k in available)
            )

        entry = self._ensemble_config[model_key]
        self._gauss_transform = entry.get("gauss_transform", True)
        self._ew_bounds = tuple(entry.get("ew_bounds", [0, 3.4]))
        self._gauss_prior_bounds = tuple(entry.get("gauss_prior_bounds", [-5, 5]))
        self.model_key = model_key

        print(f"Loading ensemble '{model_key}' ({len(entry['files'])} models)...")
        paths = fetch_ensemble_models(entry)
        self.posteriors = [
            load_posterior_from_path(p)
            for p in tqdm(paths, desc="Loading models", unit="model")
        ]

        print(f"[Ensemble Loaded] {model_key}")
        print(f"  Models: {len(self.posteriors)}")
        print(f"  Gaussian transform: {self._gauss_transform}")
        if self._gauss_transform:
            print(f"  EW bounds: {self._ew_bounds}")

    def _sample_ensemble(self, data, num_samples, percentiles, kde_resolution):
        if self.posteriors is None:
            raise RuntimeError(
                "No ensemble loaded. Call load_model('<model_key>') first."
            )

        x_obs_tensor = self._to_tensor(data)
        n_models = len(self.posteriors)
        n_per_model = num_samples // n_models

        all_samples = []
        for post in self.posteriors:
            s = post.sample_batched((n_per_model,), x=x_obs_tensor)
            s = s.squeeze(-1).T
            all_samples.append(s)

        ensemble = torch.cat(all_samples, dim=1).numpy()

        if self._gauss_transform:
            ensemble = cdf_transform(ensemble, self._ew_bounds)

        max_ew = []
        for dist in ensemble:
            _, peak = make_kde(dist, self._ew_bounds, kde_resolution)
            max_ew.append(peak)

        pct_values = np.percentile(ensemble, q=list(percentiles), axis=1)

        col_names = [f"p{int(p)}_ew" for p in percentiles]
        summary_data = {"max_ew": np.array(max_ew)}
        for name, vals in zip(col_names, pct_values):
            summary_data[name] = vals

        self.post_samples = ensemble
        self.summary_df = pd.DataFrame(summary_data)
        return ensemble, self.summary_df

    @staticmethod
    def _to_tensor(data):
        if isinstance(data, torch.Tensor):
            return data.float()
        if isinstance(data, pd.DataFrame):
            data = data.values
        return torch.as_tensor(data, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Unified sample() entry point
    # ------------------------------------------------------------------

    def sample(self, data, num_samples=10000, percentiles=(16, 50, 84), kde_resolution=1000):
        """Run inference and return posterior samples + summary statistics.

        In ensemble mode returns (samples_array, summary_df).
        In single-model mode sets self.post_samples / self.summary_df (legacy API).

        Args:
            data: Input observations — numpy array, torch tensor, or DataFrame.
            num_samples: Total posterior samples to draw. In ensemble mode these
                         are split evenly across all models in the ensemble.
            percentiles: Percentiles to compute for the summary DataFrame.
                         Default (16, 50, 84); pass e.g. (2.5, 16, 50, 84, 97.5)
                         for wider credible intervals.
            kde_resolution: Number of grid points for the KDE mode estimate.
                            Only used in ensemble mode.

        Returns:
            Ensemble mode: (samples_array, summary_df)
            Single-model mode: None (results in self.post_samples, self.summary_df)
        """
        if self._mode == "ensemble":
            return self._sample_ensemble(data, num_samples, percentiles, kde_resolution)
        else:
            self.predict(data, num_samples)
            self.generate_summary(self.post_samples)

    # ------------------------------------------------------------------
    # Single-model mode methods (legacy, unchanged)
    # ------------------------------------------------------------------

    def grab_model(self, file):
        self.posterior = load_model(file)

    def check_model_is_loaded(self):
        if self.posterior is None:
            raise RuntimeError("Posterior Model failed to load. Check file path!")

    def check_data_type(self, data):
        if isinstance(data, torch.Tensor):
            print("This is a PyTorch tensor")
        elif isinstance(data, np.ndarray):
            print("This is a NumPy array, converting to PyTorch Tensor")
            data = torch.as_tensor(data, dtype=torch.float32).to(config.DEVICE)
        elif isinstance(data, pd.DataFrame):
            print("This is a DataFrame, converting to PyTorch Tensor")
            data = data.values
            data = torch.as_tensor(data, dtype=torch.float32).to(config.DEVICE)
        return data

    def predict(self, data, num_samples):
        x_obs = self.check_data_type(data)
        self.validate_shape(x_obs)
        self.data = x_obs.numpy()
        samples = sample_from_posterior(self.posterior, x_obs, num_samples)
        self.post_samples = samples.numpy()

    def generate_summary(self, samples):
        l2pt5, l16, med, u84, u97pt5 = grab_percentiles_from_posterior(samples)
        percentile_data = dict(zip(config.PERCENTILE_NAMES, [l2pt5, l16, med, u84, u97pt5]))
        self.summary_df = pd.DataFrame(percentile_data)

    def _return_quantities(self):
        return self.data, self.post_samples, self.summary_df

    def load_model_schema(self, schema_file=None):
        if schema_file is None:
            model_dir = Path(__file__).parent / "models"
            schema_file = model_dir / "model_inputs.yaml"
        with open(schema_file) as f:
            self.model_schema = yaml.safe_load(f)

    def init_display_message(self, model_key):
        schema = self.model_schema[model_key]
        expected_features = schema["input_features"]
        expected_dim = schema["input_dim"]
        feature_list = ", ".join(expected_features)
        message = (
            f"\n[Model Loaded Successfully]\n"
            f"  • Model key detected: {model_key}\n"
            f"  • Expected input dimensionality: {expected_dim}\n"
            f"  • Required input feature order:\n"
            f"    [{feature_list}]\n\n"
            f"Please ensure that your input data columns follow this exact order\n"
            f"and have shape (N, {expected_dim}) before running inference.\n"
        )
        print(message)

    def validate_shape(self, data):
        schema = self.model_schema[self.model_key]
        expected_dim = schema["input_dim"]
        if data.shape[1] != expected_dim:
            raise ValueError(
                f"Expected input dim {expected_dim}, got {data.shape[1]}. "
                "Please check input data."
            )
