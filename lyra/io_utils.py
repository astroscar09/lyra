import pickle, io, torch
from pathlib import Path
from importlib.resources import files, as_file


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
