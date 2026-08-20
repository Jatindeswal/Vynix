"""
PROJECT VYNIX — Few-Shot Human-Object Interaction (HOI) Detector
================================================================
Four-stage pipeline:  Node Extraction → Scene Graph Math → VLM Alignment → Vynix Logic Gate.
Evaluates two benchmark images and produces annotated visual proof outputs.
CPU-only.  No CUDA required.
"""
import math, os, cv2, torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

# ── §1  NODE EXTRACTION ("The Eyes") ─────────────────────────────────────────
# YOLOv8-nano localises the human agent (COCO class 0) and object patient
# (class 41 = cup).  Returns the first detection per class above threshold.

def extract_nodes(image_path, person_cls=0, object_cls=41, conf=0.25):
    """Return (person_box, object_box) as [x1,y1,x2,y2] or None."""
    model = YOLO("yolov8n.pt")
    results = model(image_path, device="cpu", verbose=False)
    pbox, obox = None, None
    for r in results:
        for i in range(len(r.boxes)):
            c, sc = int(r.boxes.cls[i].item()), float(r.boxes.conf[i].item())
            if sc < conf: continue
            xy = r.boxes.xyxy[i].tolist()
            if c == person_cls and pbox is None: pbox = xy
            elif c == object_cls and obox is None: obox = xy
            if pbox and obox: return pbox, obox
    return pbox, obox

# ── §2  SCENE GRAPH MATH ("The Brain") ──────────────────────────────────────
# Pure geometric functions — no model dependencies.  IoU = 0.0 is the critical
# sentinel that triggers the Vynix Logic Gate override (see §4).

def centroid(box):
    """Geometric centre: ((x1+x2)/2, (y1+y2)/2)."""
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def euclidean(a, b):
    """L2 distance between two 2-D points."""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def iou(a, b):
    """Intersection-over-Union ∈ [0,1].  Returns 0.0 for disjoint boxes."""
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/union if union > 0 else 0.0

def union_box(a, b):
    """Minimum enclosing rectangle of two boxes — the VLM crop region."""
    return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]

# ── §3  VLM ALIGNMENT ("The Translator") ────────────────────────────────────
# CLIP (Radford et al., 2021) scores the union-box crop against two competing
# interaction hypotheses via contrastive image-text alignment.

PROMPTS = ["a person holding a cup",
           "a person standing near a cup but not touching it"]

def vlm_classify(image_path, ubox, prompts=PROMPTS):
    """Crop union box, run CLIP, return {prompt: softmax_prob}."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    x1,y1,x2,y2 = max(0,int(ubox[0])), max(0,int(ubox[1])), min(w,int(ubox[2])), min(h,int(ubox[3]))
    crop = Image.fromarray(cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cpu").eval()
    inputs = proc(text=prompts, images=crop, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image          # shape: (1, 2)
    probs = torch.softmax(logits, dim=1).squeeze(0).tolist()  # softmax → probabilities
    return dict(zip(prompts, probs))

# ── §4  THE VYNIX LOGIC GATE ────────────────────────────────────────────────
#   V(·) = { "No Interaction"   if argmax(VLM) ∈ contact ∧ IoU = 0
#           { argmax(VLM)        otherwise
#
# Rationale:  Contact interactions (holding, carrying) require non-zero spatial
# overlap.  If IoU = 0, the bounding boxes are disjoint → physical contact is
# geometrically impossible.  The gate overrides VLM hallucinations with a hard
# geometric veto, eliminating false-positive contact predictions.

def vynix_gate(vlm_probs, iou_val):
    """Fuse VLM semantics with geometric evidence.  Returns final verdict."""
    top = max(vlm_probs, key=vlm_probs.get)
    # Gate fires when VLM predicts contact ("holding") but IoU proves no overlap
    if "holding" in top.lower() and iou_val == 0.0:
        return "a person standing near a cup but not touching it"  # OVERRIDE
    return top  # Trust VLM

# ── §5  VISUAL PROOF RENDERER ───────────────────────────────────────────────
# Draws bounding boxes, IoU, and the Vynix verdict on the image for viva slides.

def render_proof(image_path, pbox, obox, iou_val, verdict, out_path):
    """Annotate image with boxes + metrics and save as visual proof."""
    img = cv2.imread(image_path)
    # Person box — Blue (BGR: 255,120,50)
    cv2.rectangle(img, (int(pbox[0]),int(pbox[1])), (int(pbox[2]),int(pbox[3])), (255,120,50), 3)
    cv2.putText(img, "Person", (int(pbox[0]),int(pbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,120,50), 2)
    # Object box — Green (BGR: 50,220,50)
    cv2.rectangle(img, (int(obox[0]),int(obox[1])), (int(obox[2]),int(obox[3])), (50,220,50), 3)
    cv2.putText(img, "Cup", (int(obox[0]),int(obox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
    # Overlay banner — IoU + Verdict
    cv2.rectangle(img, (0,0), (img.shape[1], 60), (30,30,30), -1)  # dark banner
    cv2.putText(img, f"IoU: {iou_val*100:.1f}%", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(img, f"Verdict: {verdict}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imwrite(out_path, img)
    print(f"  [PROOF] Saved: {out_path}")

# ── §6  BENCHMARK PIPELINE ──────────────────────────────────────────────────
# Runs the full four-stage pipeline on a single image and returns all metrics.

def run_pipeline(image_path, out_path, label):
    """Execute Stages 1–4 on one image, render proof, return results dict."""
    print(f"\n{'='*60}\n  [{label}] Processing: {os.path.basename(image_path)}\n{'='*60}")

    # Stage 1 — Node Extraction
    pbox, obox = extract_nodes(image_path)
    if not pbox or not obox:
        print("  ⚠ Detection failed — one or both nodes missing.")
        return None

    # Stage 2 — Scene Graph Math
    pc, oc = centroid(pbox), centroid(obox)
    dist = euclidean(pc, oc)
    iou_val = iou(pbox, obox)
    print(f"  Person: {[round(v,1) for v in pbox]}  Centroid: ({pc[0]:.0f},{pc[1]:.0f})")
    print(f"  Cup:    {[round(v,1) for v in obox]}  Centroid: ({oc[0]:.0f},{oc[1]:.0f})")
    print(f"  Distance: {dist:.1f}px   IoU: {iou_val:.4f}")

    # Stage 3 — VLM Alignment
    ubox = union_box(pbox, obox)
    probs = vlm_classify(image_path, ubox)
    for p, v in probs.items(): print(f"  CLIP  \"{p}\": {v:.4f}")

    # Stage 4 — Vynix Logic Gate
    verdict = vynix_gate(probs, iou_val)
    gate_status = "OVERRIDE" if ("holding" in max(probs, key=probs.get).lower() and iou_val == 0.0) else "PASS"
    print(f"  Gate: {gate_status}  →  Verdict: {verdict}")

    # Visual Proof
    render_proof(image_path, pbox, obox, iou_val, verdict, out_path)
    return {"person": pbox, "cup": obox, "iou": iou_val, "distance": dist,
            "vlm": probs, "verdict": verdict, "gate": gate_status}

# ── §7  MAIN — TWO-IMAGE BENCHMARK ──────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    benchmarks = [
        # (input image,                       output proof,                         label)
        (os.path.join(base, "test_image.jpg"),  os.path.join(base, "vynix_proof_holding.jpg"),      "HOLDING"),
        (os.path.join(base, "test_image1.jpg"), os.path.join(base, "vynix_proof_not_holding.jpg"),   "NOT-HOLDING"),
    ]
    print("\n  PROJECT VYNIX — Two-Image Benchmark\n")
    results = {}
    for img_path, out_path, label in benchmarks:
        if not os.path.exists(img_path):
            print(f"  ⚠ Skipping {os.path.basename(img_path)} — file not found."); continue
        results[label] = run_pipeline(img_path, out_path, label)

    # Summary table
    print(f"\n{'='*60}\n  BENCHMARK SUMMARY\n{'='*60}")
    for label, r in results.items():
        if r: print(f"  [{label}]  IoU={r['iou']:.4f}  Gate={r['gate']}  Verdict={r['verdict']}")
    print()
