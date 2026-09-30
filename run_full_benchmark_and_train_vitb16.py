import os
import sys
import subprocess
import time

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

def run_step(cmd, desc):
    print("=" * 80)
    print(f"  STARTING: {desc}")
    print(f"  Command: {cmd}")
    print("=" * 80)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    if ret.returncode != 0:
        print(f"[ERROR] {desc} failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)
    print(f"[DONE] Completed {desc} in {dt/60:.2f} minutes.\n")

if __name__ == "__main__":
    dataset_dir = r"E:\Dataset"
    
    # -- Phase 1: Full 9,658 Test Images Benchmark with YOLOv8-medium + Soft Gate --
    print("\n[PHASE 1/3] Benchmarking Vynix (YOLOv8-medium + Soft Geometric Veto Gate) on 9,658 images...")
    cmd_eval_m = (
        f'python test_vynix_pretrained.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--model-dir "saved_models" '
        f'--detector-model "yolov8m.pt" '
        f'--clip-model "openai/clip-vit-base-patch32"'
    )
    run_step(cmd_eval_m, "Phase 1: Full Test Set Benchmark (YOLOv8m + Soft Veto Gate)")
    
    # -- Phase 2: Full Dataset Training with ViT-B/16 + YOLOv8-medium --
    print("\n[PHASE 2/3] Extracting Cache & Training Vynix Adapter with ViT-B/16 (196 tokens) + YOLOv8-medium...")
    os.makedirs("saved_models_vitb16", exist_ok=True)
    cmd_train_b16 = (
        f'python train_vynix_full.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--output-dir "saved_models_vitb16" '
        f'--detector-model "yolov8m.pt" '
        f'--clip-model "openai/clip-vit-base-patch16" '
        f'--epochs 15 '
        f'--batch-size 256'
    )
    run_step(cmd_train_b16, "Phase 2: Training ViT-B/16 Adapter on 38,118 images")
    
    # -- Phase 3: Benchmark ViT-B/16 on Full Test Set --
    print("\n[PHASE 3/3] Benchmarking ViT-B/16 + YOLOv8-medium on Full Test Set...")
    cmd_eval_b16 = (
        f'python test_vynix_pretrained.py '
        f'--dataset-dir "{dataset_dir}" '
        f'--model-dir "saved_models_vitb16" '
        f'--detector-model "yolov8m.pt" '
        f'--clip-model "openai/clip-vit-base-patch16"'
    )
    run_step(cmd_eval_b16, "Phase 3: ViT-B/16 Full Benchmark Evaluation")
    
    print("\nALL PHASES COMPLETED SUCCESSFULLY!")
