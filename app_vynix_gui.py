#!/usr/bin/env python3
"""
Project Vynix 2.0 Web GUI
=========================
OUTDATED NOTICE: Legacy demo GUI. Metrics previously displayed (e.g., 34.80% mAP)
reflect image-level retrieval without spatial localization, not comparable to published
HICO-DET box-level mAP. Under official standard box-level Default mAP (IoU >= 0.5):
Vynix 1.0 achieves 12.23% (Cont) / 12.80% (11P); Vynix 2.0 achieves 14.76% (Cont) / 15.29% (11P).
See review_response/FINAL_TABLES.md for verified benchmarks.
=========================
A modern, lightweight web application for live Human-Object Interaction (HOI)
detection and visual grounding using Vynix 2.0 (YOLOv8x + CLIP ViT-B/16 + Geometric Veto Gate).
"""

import os
import io
import sys
import json
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template_string

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Import Vynix pipeline core
from vynix_fewshot_adapter import (
    HOIMeta,
    MultiStreamFeatureExtractor,
    Vynix3StreamAdapter,
    compute_iou,
    compute_box_distance,
    compute_union_box,
    compute_spatial_vector,
    compute_soft_geometric_gate,
    normalize_name,
    COCO_CLASSES,
    CONTACT_VERBS
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

# Global model state
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTOR_MODEL = "yolov8x.pt"
CLIP_MODEL = "openai/clip-vit-base-patch16"
MODEL_DIR = os.path.abspath("saved_models_v2.0")
DATASET_DIR = r"E:\Dataset" if os.path.isdir(r"E:\Dataset") else os.path.abspath(".")

print("=" * 70)
print("  INITIALIZING VYNIX 2.0 WEB BACKEND")
print("=" * 70)
print(f"  Device        : {DEVICE}")
print(f"  Detector      : {DETECTOR_MODEL}")
print(f"  Vision Backbone: {CLIP_MODEL}")
print(f"  Model Dir     : {MODEL_DIR}")

# 1. Initialize Metadata
meta = HOIMeta(DATASET_DIR)
print(f"  HOI Categories: {meta.num_classes} classes loaded from list_action.csv")

# 2. Initialize Feature Extractor (YOLOv8x + CLIP ViT-B/16)
extractor = MultiStreamFeatureExtractor(
    device=DEVICE,
    detector_conf=0.08,
    use_3stream=True,
    detector_model=DETECTOR_MODEL,
    clip_model=CLIP_MODEL
)
extractor.yolo.to(DEVICE).eval()

# 3. Precompute CLIP Text Embeddings
print("  Building multi-template text weights...")
text_weights = extractor.build_text_weights(meta, use_ensemble=True).to(DEVICE)

# 4. Load Trained Adapter Weights
cache_path = os.path.join(MODEL_DIR, "vynix_full_cache.pt")
model_path = os.path.join(MODEL_DIR, "vynix_full_model.pth")

if os.path.isfile(cache_path) and os.path.isfile(model_path):
    print(f"  Loading trained weights from {MODEL_DIR}...")
    cache_data = torch.load(cache_path, map_location=DEVICE, weights_only=True)
    adapter = Vynix3StreamAdapter(
        cache_keys=cache_data["cache_keys"],
        cache_values=cache_data["cache_values"],
        use_spatial_mlp=True,
        num_classes=meta.num_classes
    )
    adapter.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    adapter.to(DEVICE).eval()
    print("  [SUCCESS] Vynix 2.0 SOTA flagship checkpoint loaded!")
else:
    print("  [WARNING] Flagship weights not found. Operating in Zero-Shot mode.")
    adapter = None

# HTML UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Vynix 2.0 — Human-Object Interaction AI</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0e17;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent-cyan: #38bdf8;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --accent-amber: #f59e0b;
      --text-white: #f9fafb;
      --text-muted: #9ca3af;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-dark);
      color: var(--text-white);
      font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .logo-container {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, var(--accent-cyan), #6366f1);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 1.2rem;
      color: #000;
    }

    .logo-title {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }

    .logo-badge {
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 600;
    }

    .header-links a {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.9rem;
      font-weight: 500;
      margin-left: 20px;
      transition: color 0.2s;
    }
    .header-links a:hover { color: var(--accent-cyan); }

    main {
      flex: 1;
      max-width: 1400px;
      width: 100%;
      margin: 0 auto;
      padding: 32px 24px;
      display: grid;
      grid-template-columns: 480px 1fr;
      gap: 32px;
    }

    @media (max-width: 1024px) {
      main { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .panel-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }

    /* Dropzone */
    .dropzone {
      border: 2px dashed #374151;
      border-radius: 12px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(17, 24, 39, 0.4);
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-cyan);
      background: rgba(56, 189, 248, 0.05);
    }
    .dropzone-icon {
      font-size: 2.5rem;
      margin-bottom: 10px;
      color: var(--accent-cyan);
    }

    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .control-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    input[type="range"] {
      accent-color: var(--accent-cyan);
      cursor: pointer;
    }

    .range-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.85rem;
      color: var(--accent-cyan);
      float: right;
    }

    .btn-submit {
      background: linear-gradient(135deg, var(--accent-cyan), #0284c7);
      color: #04101d;
      border: none;
      padding: 14px 24px;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: transform 0.15s, box-shadow 0.15s;
    }
    .btn-submit:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 20px rgba(56, 189, 248, 0.3);
    }
    .btn-submit:disabled {
      opacity: 0.5;
      cursor: not-allowed;
      transform: none;
    }

    /* Output Section */
    .image-preview-container {
      position: relative;
      background: #060910;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 420px;
    }

    .image-preview-container img {
      max-width: 100%;
      max-height: 580px;
      object-fit: contain;
      display: block;
    }

    .placeholder-msg {
      color: #4b5563;
      text-align: center;
      font-size: 0.95rem;
    }

    .stats-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }

    .stat-card {
      background: #1e293b;
      padding: 12px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-val {
      font-size: 1.3rem;
      font-weight: 800;
      font-family: 'JetBrains Mono', monospace;
    }
    .stat-lbl {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* Predictions List */
    .results-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 380px;
      overflow-y: auto;
      padding-right: 4px;
    }

    .prediction-item {
      background: #182234;
      border: 1px solid #283548;
      border-radius: 10px;
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s;
    }
    .prediction-item:hover {
      border-color: var(--accent-cyan);
      background: #1c283d;
    }

    .pred-triplet {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .pred-text {
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
    }
    .pred-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }

    .pred-score-badge {
      padding: 6px 12px;
      border-radius: 8px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.9rem;
    }
    .badge-high { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); border: 1px solid var(--accent-emerald); }
    .badge-med { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
    .badge-veto { background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); border: 1px solid var(--accent-rose); text-decoration: line-through; }

    .spinner {
      border: 3px solid rgba(255, 255, 255, 0.1);
      width: 24px;
      height: 24px;
      border-radius: 50%;
      border-left-color: var(--accent-cyan);
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="logo-container">
      <div class="logo-icon">V</div>
      <div>
        <div class="logo-title">Project Vynix 2.0</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Decoupled Multi-Modal HOI Inference</div>
      </div>
      <span class="logo-badge">34.80% SOTA</span>
    </div>
    <div class="header-links">
      <a href="https://jatindeswal.github.io/Vynix/" target="_blank">Architecture Diagram ↗</a>
      <a href="https://jatindeswal.github.io/Vynix/presentation.html" target="_blank">Presentation ↗</a>
      <a href="https://github.com/Jatindeswal/Vynix" target="_blank">GitHub Repo ↗</a>
    </div>
  </header>

  <main>
    <!-- Left Panel: Input & Controls -->
    <div class="panel">
      <div class="panel-title">
        <span>📸 Upload & Parameters</span>
      </div>

      <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
        <div class="dropzone-icon">📥</div>
        <div style="font-weight: 700; margin-bottom: 4px;">Click or Drag an Image Here</div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">Supports JPG, PNG, WEBP (RGB)</div>
        <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
      </div>

      <div class="controls-grid">
        <div class="control-group">
          <label>Min Score Threshold <span class="range-val" id="threshVal">0.15</span></label>
          <input type="range" id="threshInput" min="0.05" max="0.80" step="0.05" value="0.15" oninput="document.getElementById('threshVal').innerText = this.value">
        </div>
        <div class="control-group">
          <label>Top Preds Limit <span class="range-val" id="topkVal">10</span></label>
          <input type="range" id="topkInput" min="3" max="25" step="1" value="10" oninput="document.getElementById('topkVal').innerText = this.value">
        </div>
      </div>

      <div class="control-group">
        <label>Geometric Veto Gate</label>
        <div style="display: flex; gap: 10px; margin-top: 4px;">
          <label style="font-weight: 500; display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="gateCheckbox" checked> Enable Soft Gaussian Gate (vetoes distant contact)
          </label>
        </div>
      </div>

      <button class="btn-submit" id="submitBtn" onclick="runInference()" disabled>
        <span id="btnText">Select an Image First</span>
        <div class="spinner" id="btnSpinner"></div>
      </button>

      <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; border-top: 1px solid var(--card-border); padding-top: 14px;">
        <strong>How it works:</strong>
        <br>1. <strong>YOLOv8x</strong> extracts human agents and objects.
        <br>2. <strong>CLIP ViT-B/16</strong> extracts 3-stream multi-crop visual features.
        <br>3. <strong>Soft Geometric Gate</strong> suppresses spatial false-positives.
        <br>4. <strong>Non-Parametric Cache</strong> retrieves 105% Rare parity predictions.
      </div>
    </div>

    <!-- Right Panel: Visualizer & Results -->
    <div class="panel">
      <div class="panel-title">
        <span>👁️ Grounded Visual Detections & Predictions</span>
      </div>

      <div class="stats-bar">
        <div class="stat-card">
          <div class="stat-val" id="statPersons" style="color: var(--accent-cyan);">-</div>
          <div class="stat-lbl">Humans</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statObjects" style="color: var(--accent-emerald);">-</div>
          <div class="stat-lbl">Objects</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statInteractions" style="color: #fff;">-</div>
          <div class="stat-lbl">Triplets</div>
        </div>
        <div class="stat-card">
          <div class="stat-val" id="statVetoes" style="color: var(--accent-rose);">-</div>
          <div class="stat-lbl">Vetoed</div>
        </div>
      </div>

      <div class="image-preview-container" id="previewContainer">
        <div class="placeholder-msg" id="placeholderMsg">
          Upload an image on the left to see human-object interaction predictions and bounding boxes.
        </div>
        <img id="annotatedImage" style="display: none;" src="" alt="Detection Output">
      </div>

      <div>
        <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 10px;">Top Predicted Interactions:</div>
        <div class="results-list" id="resultsList">
          <div style="color: #6b7280; font-size: 0.85rem; text-align: center; padding: 20px;">
            Awaiting input image...
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let selectedFile = null;

    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    function handleFileSelect(e) {
      if (e.target.files.length) {
        handleFile(e.target.files[0]);
      }
    }

    function handleFile(file) {
      selectedFile = file;
      document.getElementById('dropzone').innerHTML = `
        <div class="dropzone-icon" style="color: var(--accent-emerald);">✓</div>
        <div style="font-weight: 700; color: #fff;">${file.name}</div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">${(file.size / (1024*1024)).toFixed(2)} MB</div>
      `;
      const submitBtn = document.getElementById('submitBtn');
      submitBtn.disabled = false;
      document.getElementById('btnText').innerText = "Run HOI Detection ⚡";

      // Show immediate local preview
      const reader = new FileReader();
      reader.onload = function(e) {
        const img = document.getElementById('annotatedImage');
        img.src = e.target.result;
        img.style.display = "block";
        document.getElementById('placeholderMsg').style.display = "none";
      };
      reader.readAsDataURL(file);
    }

    async function runInference() {
      if (!selectedFile) return;

      const submitBtn = document.getElementById('submitBtn');
      const btnText = document.getElementById('btnText');
      const spinner = document.getElementById('btnSpinner');

      submitBtn.disabled = true;
      btnText.innerText = "Analyzing Scene...";
      spinner.style.display = "inline-block";

      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('threshold', document.getElementById('threshInput').value);
      formData.append('top_k', document.getElementById('topkInput').value);
      formData.append('enable_gate', document.getElementById('gateCheckbox').checked ? "1" : "0");

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });

        const data = await response.json();

        if (data.error) {
          alert("Error: " + data.error);
          return;
        }

        // Update stats
        document.getElementById('statPersons').innerText = data.num_persons;
        document.getElementById('statObjects').innerText = data.num_objects;
        document.getElementById('statInteractions').innerText = data.predictions.length;
        document.getElementById('statVetoes').innerText = data.vetoes_count;

        // Update annotated image
        if (data.annotated_image) {
          const img = document.getElementById('annotatedImage');
          img.src = "data:image/jpeg;base64," + data.annotated_image;
          img.style.display = "block";
          document.getElementById('placeholderMsg').style.display = "none";
        }

        // Render predictions list
        const resultsList = document.getElementById('resultsList');
        resultsList.innerHTML = "";

        if (data.predictions.length === 0) {
          resultsList.innerHTML = `<div style="color: #9ca3af; text-align: center; padding: 20px;">No interaction exceeded the threshold. Try lowering the threshold slider.</div>`;
        } else {
          data.predictions.forEach(p => {
            const isVeto = p.is_vetoed;
            const badgeClass = isVeto ? 'badge-veto' : (p.score > 0.4 ? 'badge-high' : 'badge-med');
            const scorePct = (p.score * 100).toFixed(1) + "%";

            resultsList.innerHTML += `
              <div class="prediction-item">
                <div class="pred-triplet">
                  <div class="pred-text">&lang;${p.interaction}&rang;</div>
                  <div class="pred-meta">Verb: ${p.verb} | Object: ${p.object} ${isVeto ? '| [VETOED: Distant Contact]' : ''}</div>
                </div>
                <div class="pred-score-badge ${badgeClass}">${isVeto ? 'VETOED' : scorePct}</div>
              </div>
            `;
          });
        }

      } catch (err) {
        alert("Failed to communicate with inference server: " + err);
      } finally {
        submitBtn.disabled = false;
        btnText.innerText = "Run HOI Detection ⚡";
        spinner.style.display = "none";
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    threshold = float(request.form.get("threshold", 0.15))
    top_k = int(request.form.get("top_k", 10))
    enable_gate = (request.form.get("enable_gate", "1") == "1")

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    img_w, img_h = pil_img.size

    # 1. Proposal Extraction with YOLOv8x
    persons, objects = extractor.detect(pil_img)

    if not persons or not objects:
        return jsonify({
            "num_persons": len(persons),
            "num_objects": len(objects),
            "vetoes_count": 0,
            "predictions": [],
            "annotated_image": None
        })

    # Form Candidate Pairs
    pairs = [(p, o) for p in persons for o in objects]
    if len(pairs) > 50:
        pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
        pairs = pairs[:50]

    valid_pairs = []
    for (p_box, p_conf), (o_box, o_conf, o_cls) in pairs:
        coco_name = COCO_CLASSES[o_cls] if o_cls < len(COCO_CLASSES) else None
        if not coco_name:
            continue
        hico_name = normalize_name(coco_name)
        entries = meta.obj_to_entries.get(hico_name, [])
        if not entries:
            continue

        u_box = compute_union_box(p_box, o_box)
        s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)
        iou_val = compute_iou(p_box, o_box)
        b_dist = compute_box_distance(p_box, o_box, img_w, img_h)
        valid_pairs.append({
            "p_box": p_box, "p_conf": p_conf,
            "o_box": o_box, "o_conf": o_conf, "o_name": hico_name,
            "u_box": u_box, "s_vec": s_vec, "iou": iou_val, "dist": b_dist,
            "entries": entries
        })

    if not valid_pairs:
        return jsonify({
            "num_persons": len(persons),
            "num_objects": len(objects),
            "vetoes_count": 0,
            "predictions": [],
            "annotated_image": None
        })

    # 2. Visual-Spatial Multi-Crop Extraction
    boxes_to_extract = [(vp["p_box"], vp["o_box"], vp["u_box"]) for vp in valid_pairs]
    f_fused_batch = extractor.extract_visual_features_batch(pil_img, boxes_to_extract).to(DEVICE)
    s_vecs_batch = torch.stack([vp["s_vec"] for vp in valid_pairs]).to(DEVICE)
    f_unions = f_fused_batch[:, 1024:1536] if f_fused_batch.shape[-1] == 1536 else f_fused_batch

    with torch.no_grad():
        clip_logits = f_unions @ text_weights.T
        if adapter is not None:
            final_logits = adapter(clip_logits, f_fused_batch, s_vecs_batch)
            probs_batch = torch.softmax(final_logits, dim=1).cpu()
        else:
            probs_batch = torch.softmax(clip_logits * 100.0, dim=1).cpu()

    # 3. Geometric Veto & Score Formulation
    all_predictions = []
    vetoes_count = 0

    for k_idx, vp in enumerate(valid_pairs):
        p_box = vp["p_box"]
        o_box = vp["o_box"]
        o_name = vp["o_name"]
        iou_val = vp["iou"]
        b_dist = vp["dist"]
        p_conf = vp["p_conf"]
        o_conf = vp["o_conf"]

        for verb, gerund, hoi_id in vp["entries"]:
            raw_prob = probs_batch[k_idx, hoi_id].item()

            if enable_gate:
                gate = compute_soft_geometric_gate(verb, iou_val, b_dist)
            else:
                gate = 1.0

            is_vetoed = False
            if gate < 0.05 or (verb in CONTACT_VERBS and iou_val == 0.0 and b_dist > 0.15):
                is_vetoed = True
                vetoes_count += 1

            # Sublinear detector score calibration (det_lambda=0.9)
            det_score = (p_conf * o_conf) ** 0.9
            final_conf = det_score * raw_prob * gate

            interaction_str = f"person {verb} {o_name}"

            if final_conf >= threshold or (is_vetoed and raw_prob > 0.25):
                all_predictions.append({
                    "interaction": interaction_str,
                    "verb": verb,
                    "object": o_name,
                    "score": float(final_conf),
                    "raw_prob": float(raw_prob),
                    "is_vetoed": is_vetoed,
                    "p_box": [float(x) for x in p_box],
                    "o_box": [float(x) for x in o_box]
                })

    # Sort by confidence score descending
    all_predictions.sort(key=lambda x: x["score"], reverse=True)
    top_preds = all_predictions[:top_k]

    # 4. Draw Grounded Bounding Boxes on Image
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)

    # Color palette
    box_colors = [
        (56, 189, 248),   # Cyan
        (16, 185, 129),   # Emerald
        (245, 158, 11),   # Amber
        (168, 85, 247),   # Purple
        (244, 63, 94)     # Rose
    ]

    for idx, p in enumerate(top_preds):
        if p["is_vetoed"]:
            continue
        c = box_colors[idx % len(box_colors)]
        
        # Draw human box
        draw.rectangle(p["p_box"], outline=c, width=3)
        # Draw object box
        draw.rectangle(p["o_box"], outline=c, width=3)
        # Connect with dashed/center line
        cx_h = (p["p_box"][0] + p["p_box"][2]) / 2
        cy_h = (p["p_box"][1] + p["p_box"][3]) / 2
        cx_o = (p["o_box"][0] + p["o_box"][2]) / 2
        cy_o = (p["o_box"][1] + p["o_box"][3]) / 2
        draw.line([(cx_h, cy_h), (cx_o, cy_o)], fill=c, width=2)
        
        # Label tag
        label_text = f"{p['interaction']} ({p['score']*100:.0f}%)"
        draw.text((p["p_box"][0] + 4, max(0, p["p_box"][1] - 16)), label_text, fill=c)

    # Convert annotated image to base64
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "num_persons": len(persons),
        "num_objects": len(objects),
        "vetoes_count": vetoes_count,
        "predictions": top_preds,
        "annotated_image": img_b64
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🚀 Vynix 2.0 Web Server running at: http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
