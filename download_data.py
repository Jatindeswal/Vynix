from huggingface_hub import hf_hub_download
import os
os.makedirs("/app/data", exist_ok=True)
hf_hub_download(repo_id="zhimeng/hico_det", filename="data/train-00000-of-00013.parquet", repo_type="dataset", local_dir="/app")
hf_hub_download(repo_id="zhimeng/hico_det", filename="data/test-00000-of-00004.parquet", repo_type="dataset", local_dir="/app")
