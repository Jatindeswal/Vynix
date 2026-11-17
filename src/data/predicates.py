#!/usr/bin/env python3
"""
HICO-DET Verb Predicate Classifications
=======================================
Single source of truth for:
- 59 HICO-DET physical contact verbs
- 58 HICO-DET non-contact / unconstrained verbs
- Total: 117 HICO-DET verb predicates
- Backward compatibility with legacy 64-entry CONTACT_VERBS set
"""

from typing import FrozenSet, Set

# The 59 canonical HICO-DET verbs classified as physical-contact interactions
HICO_CONTACT_VERBS: FrozenSet[str] = frozenset([
    "board", "brush_with", "carry", "catch", "control", "cut", "cut_with",
    "drag", "dribble", "drink_with", "drive", "dry", "eat", "fill", "flip",
    "fly", "grind", "groom", "hit", "hold", "hop_on", "hose", "hug", "kick",
    "kiss", "lick", "lie_on", "lift", "load", "milk", "open", "operate",
    "pack", "peel", "pet", "pick", "pick_up", "pour", "pull", "push", "ride",
    "row", "sail", "shear", "sip", "sit_at", "sit_on", "spin", "squeeze",
    "stab", "stand_on", "straddle", "swing", "tie", "type_on", "wash", "wear",
    "wield", "zip",
])

# The 58 canonical HICO-DET verbs classified as non-contact / unconstrained interactions
HICO_NON_CONTACT_VERBS: FrozenSet[str] = frozenset([
    "adjust", "assemble", "block", "blow", "break", "buy", "chase", "check",
    "clean", "cook", "direct", "eat_at", "exit", "feed", "flush", "greet",
    "herd", "hunt", "inspect", "install", "jump", "lasso", "launch", "light",
    "lose", "make", "move", "no_interaction", "paint", "park", "pay", "point",
    "race", "read", "release", "repair", "run", "scratch", "serve", "set",
    "sign", "slide", "smell", "stand_under", "stick", "stir", "stop_at",
    "tag", "talk_on", "teach", "text_on", "throw", "toast", "train", "turn",
    "walk", "watch", "wave",
])

# 5 legacy aliases present in original codebase but not appearing as vname in HICO-DET list_action.csv
LEGACY_CONTACT_ALIASES: FrozenSet[str] = frozenset([
    "grab", "mount", "play_with", "taste", "touch",
])

# Full backwards-compatible set matching vynix_fewshot_adapter.py (64 strings)
CONTACT_VERBS: Set[str] = set(HICO_CONTACT_VERBS | LEGACY_CONTACT_ALIASES)
NON_CONTACT_VERBS: Set[str] = set(HICO_NON_CONTACT_VERBS)


def is_contact_verb(verb: str) -> bool:
    """Returns True if the verb requires physical contact (and thus spatial gating)."""
    return verb.strip().lower() in CONTACT_VERBS


# Sanity assertions
assert len(HICO_CONTACT_VERBS) == 59, f"Expected 59 contact verbs, got {len(HICO_CONTACT_VERBS)}"
assert len(HICO_NON_CONTACT_VERBS) == 58, f"Expected 58 non-contact verbs, got {len(HICO_NON_CONTACT_VERBS)}"
assert len(HICO_CONTACT_VERBS & HICO_NON_CONTACT_VERBS) == 0, "Contact and non-contact sets must be disjoint"
assert len(HICO_CONTACT_VERBS | HICO_NON_CONTACT_VERBS) == 117, f"Expected 117 HICO verbs, got {len(HICO_CONTACT_VERBS | HICO_NON_CONTACT_VERBS)}"
assert len(HICO_CONTACT_VERBS) + len(HICO_NON_CONTACT_VERBS) == 117, "Sum of contact + non-contact must equal 117"
assert len(CONTACT_VERBS) == 64, f"Expected 64 contact verbs for backward compatibility, got {len(CONTACT_VERBS)}"
