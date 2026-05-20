import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from lyra.io_utils import fmt_size, get_remote_size, is_cached_hf, confirm_download

HF_DATASET_REPO = "chavezoscar009/lyra_data"
TRAINING_DATA_FILENAME = "lyra_training_data.fits.gz"
FULL_LAE_ML_FILENAME = "Full_lyra_ML_Set_LAEs_Only.fits.gz"


def download_training_data(dest_dir=None) -> Path:
    """Download the EW-uniform training subset used to train the Lyra SBI models.

    This is a subset of the full LAE ML data engineered to be uniform in
    Lyman-alpha equivalent width so no EW bin dominates training.

    Args:
        dest_dir: Directory to save the file. Defaults to current directory.

    Returns:
        Path to the downloaded FITS file.
    """
    _maybe_confirm(TRAINING_DATA_FILENAME, "training data subset (EW-uniform)")
    cached = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=TRAINING_DATA_FILENAME,
        repo_type="dataset",
    )
    return _deliver(cached, TRAINING_DATA_FILENAME, dest_dir)


def download_full_LAE_ML_data(dest_dir=None) -> Path:
    """Download the full Lyman-alpha emitter ML dataset (1.8M galaxies).

    Contains all galaxy properties used as inputs to Lyra, plus Lyman-alpha
    equivalent widths. The training set is a uniform-EW subset of this catalog.

    Args:
        dest_dir: Directory to save the file. Defaults to current directory.

    Returns:
        Path to the downloaded FITS file.
    """
    _maybe_confirm(FULL_LAE_ML_FILENAME, "full LAE ML dataset")
    cached = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=FULL_LAE_ML_FILENAME,
        repo_type="dataset",
    )
    return _deliver(cached, FULL_LAE_ML_FILENAME, dest_dir)


def _maybe_confirm(filename, label):
    """Show size and ask for confirmation if the file isn't already cached."""
    if not is_cached_hf(HF_DATASET_REPO, filename, repo_type="dataset"):
        size = get_remote_size(HF_DATASET_REPO, filename, repo_type="dataset")
        desc = f"Downloading {label} ('{filename}') from '{HF_DATASET_REPO}'"
        if not confirm_download(desc, size):
            raise RuntimeError("Download cancelled by user.")


def _deliver(cached_path, filename, dest_dir):
    """Copy a cached HF file to dest_dir with a progress bar."""
    dest_dir = Path(dest_dir) if dest_dir else Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename

    total = Path(cached_path).stat().st_size
    chunk = 1024 * 1024  # 1 MB

    with open(cached_path, "rb") as src, open(dest, "wb") as dst:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Saving {filename}") as pbar:
            while buf := src.read(chunk):
                dst.write(buf)
                pbar.update(len(buf))

    print(f"Saved to: {dest}")
    return dest
