from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="zhimeng/hico_det", filename="list_action.csv", repo_type="dataset", local_dir="/app")
