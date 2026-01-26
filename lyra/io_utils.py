import pickle, io, torch
from pathlib import Path
from importlib.resources import files, as_file
from huggingface_hub import hf_hub_download

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
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)

def load_inference(path, device="cpu"):
    with open(path, "rb") as f:
        inference = CPU_Unpickler(f).load()
    
    inference._device = device
    #inference._neural_net = inference._neural_net.to(device)
    posterior = inference.build_posterior()
    return posterior, inference


def load_trained_model(path, map_location="cpu"):
    
    with as_file(path) as f:
        if not f.exists():
            raise FileNotFoundError(f"Model not found, check filepath: {model_name}")
        posterior, inference = load_inference(f, device=map_location)
        return posterior, inference


def grab_model_file(file):
    stem = Path(file).stem
    return stem
