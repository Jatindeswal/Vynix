import os
import sys
import subprocess
import time
import json

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def run_step(cmd, desc):
    print("=" * 80)
    print(f"  [STARTING] {desc}")
    print(f"  Command: {cmd}")
    print("=" * 80)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"\n[ERROR] {desc} failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)
    print(f"\n[DONE] Completed {desc} in {dt/60:.2f} minutes.\n")
    return dt

if __name__ == "__main__":
    dataset_dir = r"E:\Dataset"
    output_dir = "saved_models_yolov8x_vitb16"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  PROJECT VYNIX — OPTION 1: RETRAIN ADAPTER ON YOLOv8x PROPOSALS")
    print("=" * 80)

    # 1. Feature Extraction & Adapter Training on YOLOv8x
    train_cmd = (
        f'python train_vynix_full.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--output-dir "{output_dir}" '
        f'--detector-model "yolov8x.pt" '
        f'--clip-model "openai/clip-vit-base-patch16" '
        f'--epochs 15 '
        f'--batch-size 64'
    )
    t_train = run_step(train_cmd, "Phase 1: YOLOv8x Cache Extraction & Adapter Training (15 Epochs)")

    # 2. Benchmark on 300-Image Controlled Validation Slice
    print("\n[VALIDATION] Testing on 300-image slice to verify representation alignment...")
    val_cmd = (
        f'python test_vynix_pretrained.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--model-dir "{output_dir}" '
        f'--detector-model "yolov8x.pt" '
        f'--clip-model "openai/clip-vit-base-patch16" '
        f'--det-lambda 0.9 '
        f'--obj-gamma 0.1 '
        f'--no-adaptive-alpha '
        f'--limit 300'
    )
    run_step(val_cmd, "Phase 2: 300-Image Validation Slice Benchmark")

    # 3. Full 9,658 Test Images Benchmark
    print("\n[FLAGSHIP EVALUATION] Launching full 9,658 test set benchmark...")
    eval_cmd = (
        f'python test_vynix_pretrained.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--model-dir "{output_dir}" '
        f'--detector-model "yolov8x.pt" '
        f'--clip-model "openai/clip-vit-base-patch16" '
        f'--det-lambda 0.9 '
        f'--obj-gamma 0.1 '
        f'--no-adaptive-alpha'
    )
    t_eval = run_step(eval_cmd, "Phase 3: Full 9,658 Test Set Benchmark")

    print("\n" + "=" * 80)
    print("  🏆 OPTION 1 WORKFLOW COMPLETED SUCCESSFULLY")
    print(f"  Training Time: {t_train/60:.2f} mins")
    print(f"  Evaluation Time: {t_eval/60:.2f} mins")
    print("=" * 80)
