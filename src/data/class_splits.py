#!/usr/bin/env python3
"""
Standard HICO-DET Class Splits and Metadata
==========================================
Canonical single source of truth for:
- 138 Rare HOI IDs (<10 training instances in standard HICO-DET)
- 462 Non-Rare HOI IDs (>=10 training instances in standard HICO-DET)
- 80 COCO Object Classes
- 117 HICO-DET Verb Predicates
"""

from typing import FrozenSet, List, Set, Tuple

# Total counts
NUM_HOI_CLASSES: int = 600
NUM_RARE_CLASSES: int = 138
NUM_NON_RARE_CLASSES: int = 462
NUM_COCO_CLASSES: int = 80
NUM_VERB_CLASSES: int = 117

# Official 138 Rare HOI Class IDs (<10 training instances)
RARE_HOI_IDS: Tuple[int, ...] = (
    8, 22, 27, 44, 50, 55, 62, 63, 66, 70, 76, 77, 80, 83, 84, 90, 99, 100,
    104, 107, 112, 127, 135, 136, 149, 158, 165, 166, 168, 172, 179, 181,
    184, 188, 189, 192, 195, 198, 205, 206, 214, 216, 222, 227, 229, 238,
    239, 254, 255, 259, 266, 269, 274, 278, 280, 285, 288, 290, 295, 297,
    303, 305, 307, 310, 311, 317, 319, 321, 324, 327, 332, 335, 337, 344,
    345, 348, 353, 354, 355, 357, 362, 365, 366, 368, 370, 371, 372, 373,
    375, 377, 381, 382, 384, 387, 390, 395, 401, 403, 405, 406, 407, 411,
    414, 417, 421, 424, 425, 427, 429, 432, 434, 439, 443, 446, 448, 452,
    453, 459, 461, 464, 467, 471, 472, 474, 475, 478, 485, 488, 491, 501,
    509, 517, 523, 532, 545, 574, 584, 599,
)

# Official 462 Non-Rare HOI Class IDs (>=10 training instances)
NON_RARE_HOI_IDS: Tuple[int, ...] = tuple(
    i for i in range(NUM_HOI_CLASSES) if i not in set(RARE_HOI_IDS)
)

RARE_HOI_SET: FrozenSet[int] = frozenset(RARE_HOI_IDS)
NON_RARE_HOI_SET: FrozenSet[int] = frozenset(NON_RARE_HOI_IDS)

# 80 COCO Object Classes
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# 117 HICO-DET Canonical Verb Predicates
HICO_VERBS: List[str] = [
    "adjust", "assemble", "block", "blow", "board", "break", "brush_with", "buy",
    "carry", "catch", "chase", "check", "clean", "control", "cook", "cut",
    "cut_with", "direct", "drag", "dribble", "drink_with", "drive", "dry", "eat",
    "eat_at", "exit", "feed", "fill", "flip", "flush", "fly", "greet", "grind",
    "groom", "herd", "hit", "hold", "hop_on", "hose", "hug", "hunt", "inspect",
    "install", "jump", "kick", "kiss", "lasso", "launch", "lick", "lie_on",
    "lift", "light", "load", "lose", "make", "milk", "move", "no_interaction",
    "open", "operate", "pack", "paint", "park", "pay", "peel", "pet", "pick",
    "pick_up", "point", "pour", "pull", "push", "race", "read", "release",
    "repair", "ride", "row", "run", "sail", "scratch", "serve", "set", "shear",
    "sign", "sip", "sit_at", "sit_on", "slide", "smell", "spin", "squeeze",
    "stab", "stand_on", "stand_under", "stick", "stir", "stop_at", "straddle",
    "swing", "tag", "talk_on", "teach", "text_on", "throw", "tie", "toast",
    "train", "turn", "type_on", "walk", "wash", "watch", "wave", "wear", "wield",
    "zip",
]


def is_rare_class(hoi_id: int) -> bool:
    """Returns True if the HOI class belongs to the standard 138 Rare split."""
    return hoi_id in RARE_HOI_SET


# Sanity assertions
assert len(RARE_HOI_IDS) == NUM_RARE_CLASSES, f"Expected {NUM_RARE_CLASSES} rare IDs, got {len(RARE_HOI_IDS)}"
assert len(NON_RARE_HOI_IDS) == NUM_NON_RARE_CLASSES, f"Expected {NUM_NON_RARE_CLASSES} non-rare IDs, got {len(NON_RARE_HOI_IDS)}"
assert len(RARE_HOI_SET & NON_RARE_HOI_SET) == 0, "Rare and non-rare sets must be disjoint"
assert len(RARE_HOI_SET | NON_RARE_HOI_SET) == NUM_HOI_CLASSES, f"Expected union size {NUM_HOI_CLASSES}"
assert len(COCO_CLASSES) == NUM_COCO_CLASSES, f"Expected {NUM_COCO_CLASSES} COCO classes, got {len(COCO_CLASSES)}"
assert len(HICO_VERBS) == NUM_VERB_CLASSES, f"Expected {NUM_VERB_CLASSES} HICO verbs, got {len(HICO_VERBS)}"
