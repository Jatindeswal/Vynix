import os
import argparse
from huggingface_hub import hf_hub_download

def download_hico_det(dataset_dir="dataset", num_train_shards=13, num_test_shards=4, download_meta=True):
    data_dir = os.path.join(dataset_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading HICO-DET dataset to: {dataset_dir}")

    # 1. Download Action List
    if download_meta:
        print("  [1/3] Downloading list_action.csv...")
        hf_hub_download(
            repo_id="zhimeng/hico_det",
            filename="list_action.csv",
            repo_type="dataset",
            local_dir=dataset_dir
        )

    # 2. Download Training Shards
    print(f"  [2/3] Downloading {num_train_shards} training shards...")
    for i in range(num_train_shards):
        shard_name = f"data/train-{i:05d}-of-00013.parquet"
        print(f"    Fetching {shard_name}...")
        hf_hub_download(
            repo_id="zhimeng/hico_det",
            filename=shard_name,
            repo_type="dataset",
            local_dir=dataset_dir
        )

    # 3. Download Test Shards
    print(f"  [3/3] Downloading {num_test_shards} test shards...")
    for i in range(num_test_shards):
        shard_name = f"data/test-{i:05d}-of-00004.parquet"
        print(f"    Fetching {shard_name}...")
        hf_hub_download(
            repo_id="zhimeng/hico_det",
            filename=shard_name,
            repo_type="dataset",
            local_dir=dataset_dir
        )

    print("\n✓ HICO-DET dataset download complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HICO-DET Dataset Shards from HuggingFace")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Target directory to store dataset")
    parser.add_argument("--train-shards", type=int, default=13, help="Number of train shards to download (1-13)")
    parser.add_argument("--test-shards", type=int, default=4, help="Number of test shards to download (1-4)")
    args = parser.parse_args()

    download_hico_det(
        dataset_dir=args.dataset_dir,
        num_train_shards=args.train_shards,
        num_test_shards=args.test_shards
    )
