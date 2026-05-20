import pickle, io, torch, requests
from pathlib import Path
from importlib.resources import files, as_file
from huggingface_hub import hf_hub_download, hf_hub_url, try_to_load_from_cache

HF_ENSEMBLE_REPO = "chavezoscar009/lyra_models"
HF_ENSEMBLE_SUBFOLDER = "ensemble_models"


def fetch_model(model_name: str = 'full_SBI_NPE_Muv_beta.pkl'):
    """
    Download model from Hugging Face if not already present.
    Returns local file path.

    Currently Available Models are:

    - full_SBI_NPE_Muv_beta.pkl
    - full_SBI_NPE_beta_ssfr_Muv_burst.pkl
    - full_SBI_NPE_Muv_mass_beta_ssfr.pkl
    - full_SBI_NPE_Av_logU_ssfr_mass_beta_Muv_burst.pkl
    - full_SBI_NPE_metallicity_beta_logU_mass_ssfr_Muv_burst.pkl
    - full_SBI_NPE_metallicity_Av_B_delta_logU_mass_ssfr_beta_Muv_burst.pkl

    """
    return hf_hub_download(
        repo_id="chavezoscar009/lyra_models",
        filename=model_name,
    )


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)

def load_inference(path, device="cpu"):
    with open(path, "rb") as f:
        inference = CPU_Unpickler(f).load()

    inference._device = device
    posterior = inference.build_posterior()
    return posterior, inference


def load_trained_model(path, map_location="cpu"):

    with as_file(path) as f:
        if not f.exists():
            raise FileNotFoundError(f"Model not found, check filepath: {path}")
        posterior, inference = load_inference(f, device=map_location)
        return posterior, inference


def load_model(model_name: str):
    """
    Fetch and load a pre-trained model by name.

    Downloads from HuggingFace Hub if not cached locally. Falls back to the
    default model ('full_SBI_NPE_Muv_beta.pkl') if the requested file is not found.

    Args:
        model_name (str): Filename of the model (e.g., 'full_SBI_NPE_Muv_beta.pkl').

    Returns:
        posterior: The loaded SBI posterior ready for sampling.
    """
    DEFAULT_MODEL = 'full_SBI_NPE_Muv_beta.pkl'

    path = Path(fetch_model(model_name))

    if path.exists():
        posterior, _ = load_trained_model(path)
    else:
        print(f'No model file found at: {path}')
        print('Defaulting to using the default model')
        model_dir = Path(__file__).parent / 'models'
        default_model = model_dir / DEFAULT_MODEL
        print(f'Defaulting to: {default_model}')
        posterior, _ = load_trained_model(default_model)

    return posterior


def grab_model_file(file):
    stem = Path(file).stem
    return stem


# ------------------------------------------------------------------
# Download helpers (shared with downloads.py)
# ------------------------------------------------------------------

def fmt_size(size_bytes):
    """Format a byte count as a human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    return f"{size_bytes / 1e6:.1f} MB"


def get_remote_size(repo_id, filename, repo_type="model"):
    """Fetch a file's size in bytes via a HEAD request. Returns None on failure."""
    try:
        url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
        r = requests.head(url, allow_redirects=True, timeout=10)
        size = int(r.headers.get("content-length", 0))
        return size if size > 0 else None
    except Exception:
        return None


def is_cached_hf(repo_id, filename, repo_type="model"):
    """Return True if the file is already in the local HuggingFace cache."""
    result = try_to_load_from_cache(repo_id=repo_id, filename=filename, repo_type=repo_type)
    return result is not None


def confirm_download(description, size_bytes=None):
    """Prompt the user to confirm a download. Returns True if they agree.

    In non-interactive environments (scripts, CI) the prompt is skipped and
    the download proceeds automatically after printing the size info.
    """
    import sys
    size_str = f"~{fmt_size(size_bytes)}" if size_bytes else "size unknown"
    print(f"\n{description}")
    print(f"Required disk space: {size_str}")
    if not sys.stdin.isatty():
        print("(Non-interactive session — proceeding automatically)")
        return True
    answer = input("Proceed with download? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


# ------------------------------------------------------------------
# Ensemble model fetching
# ------------------------------------------------------------------

def fetch_ensemble_models(model_entry: dict) -> list:
    """Resolve all pkl files for an ensemble config entry to local paths.

    Files stored as absolute paths are loaded directly. Bare filenames are
    fetched from HuggingFace (HF_ENSEMBLE_REPO / HF_ENSEMBLE_SUBFOLDER).
    Prompts the user with total download size before fetching any new files.

    Args:
        model_entry: Config dict with a 'files' key listing filenames or paths.

    Returns:
        List of Path objects pointing to local pkl files.
    """
    paths = []
    remote_fnames = []

    for fname in model_entry["files"]:
        p = Path(fname)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Model file not found: {p}")
            paths.append(p)
        else:
            remote_fnames.append(fname)

    if remote_fnames:
        # Only count files not already in cache
        hf_paths = [f"{HF_ENSEMBLE_SUBFOLDER}/{f}" for f in remote_fnames]
        uncached = [
            (fname, hf_path)
            for fname, hf_path in zip(remote_fnames, hf_paths)
            if not is_cached_hf(HF_ENSEMBLE_REPO, hf_path)
        ]

        if uncached:
            total_size = sum(
                s for s in (
                    get_remote_size(HF_ENSEMBLE_REPO, hf_path)
                    for _, hf_path in uncached
                ) if s
            )
            desc = (
                f"Downloading {len(uncached)} model file(s) from "
                f"'{HF_ENSEMBLE_REPO}/{HF_ENSEMBLE_SUBFOLDER}'"
            )
            if not confirm_download(desc, total_size or None):
                raise RuntimeError("Download cancelled by user.")

        for fname in remote_fnames:
            local = hf_hub_download(
                repo_id=HF_ENSEMBLE_REPO,
                filename=fname,
                subfolder=HF_ENSEMBLE_SUBFOLDER,
            )
            paths.append(Path(local))

    return paths


def load_posterior_from_path(path) -> object:
    """Load a pickled SBI inference object and build its posterior."""
    with open(path, "rb") as f:
        inference = CPU_Unpickler(f).load()
    return inference.build_posterior(inference._neural_net.to("cpu"))
