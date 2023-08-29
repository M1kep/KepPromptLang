# If the models/checkpoints folder does not have test.txt, then download the model.
import os

from huggingface_hub import hf_hub_download

FILE = "v1-5-pruned-emaonly.safetensors"
REPO_ID = "runwayml/stable-diffusion-v1-5"
if not os.path.exists(f"models/checkpoints/{FILE}"):
    print("Downloading model...")
    hf_hub_download(repo_id=REPO_ID, filename=FILE, local_dir="models/checkpoints", local_dir_use_symlinks=False)
else:
    print("Model already downloaded.")
