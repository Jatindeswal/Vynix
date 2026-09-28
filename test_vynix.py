import torch
import math
from vynix_fewshot_adapter import Vynix3StreamAdapter, compute_spatial_vector
from evaluate_vynix_full import apply_vynix_gate, HallucinationTracker

def test_apply_vynix_gate():
    tracker = HallucinationTracker()

    # 1. Non-contact verb -> no veto
    assert apply_vynix_gate("look_at", 0.9, 0.0, tracker, "apple") == 0.9

    # 2. Contact verb, iou > 0 -> no veto
    assert apply_vynix_gate("hold", 0.9, 0.1, tracker, "apple") == 0.9

    # 3. Contact verb, iou == 0 -> veto
    assert apply_vynix_gate("hold", 0.9, 0.0, tracker, "apple") == 0.0

    print("test_apply_vynix_gate passed")

def test_cache_affinity_blending():
    num_classes = 5
    feature_dim = 10

    # K_cache (M, feature_dim)
    cache_keys = torch.randn(3, feature_dim)
    cache_keys = torch.nn.functional.normalize(cache_keys, dim=1)

    # cache_values (M, num_classes)
    cache_values = torch.zeros(3, num_classes)
    cache_values[0, 0] = 1.0
    cache_values[1, 1] = 1.0
    cache_values[2, 2] = 1.0

    adapter = Vynix3StreamAdapter(cache_keys, cache_values, use_spatial_mlp=False, num_classes=num_classes)
    adapter.alpha.data.fill_(0.35)
    adapter.beta.data.fill_(5.5)

    # Test query that exactly matches cache_keys[0]
    query = cache_keys[0:1] # (1, feature_dim)
    clip_logits = torch.zeros(1, num_classes)

    out = adapter(clip_logits, query)

    # affinity for match should be 1.0 -> exp(-5.5 * 0) = 1.0
    # cache_logits = [1, 0, 0, 0, 0]
    # blended_logits = clip_logits + 0.35 * cache_logits = [0.35, 0, 0, 0, 0]
    assert torch.allclose(out[0, 0], torch.tensor(0.35))

    print("test_cache_affinity_blending passed")

def test_spatial_vector():
    p_box = [10, 10, 20, 20]
    o_box = [30, 30, 40, 40]
    img_w, img_h = 100, 100

    s_vec = compute_spatial_vector(p_box, o_box, img_w, img_h)

    assert len(s_vec) == 8

    print("test_spatial_vector passed")

if __name__ == "__main__":
    test_apply_vynix_gate()
    test_cache_affinity_blending()
    test_spatial_vector()
