import os

from huggingface_hub import snapshot_download

# Create datasets/canopy folder in Open-Canopy
if not os.path.isdir("datasets"):
    os.makedirs("datasets")

# Download from huggingface
snapshot_download(
    repo_id="AI4Forest/Open-Canopy",
    repo_type="dataset",
    local_dir="datasets",
)
