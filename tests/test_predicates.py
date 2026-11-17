#!/usr/bin/env python3
"""
Unit test asserting the partition and count of HICO-DET verb predicates:
- 59 contact verbs
- 58 non-contact verbs
- Total sum == 117
- Disjointness and behavior of is_contact_verb
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.predicates import (
    CONTACT_VERBS,
    HICO_CONTACT_VERBS,
    HICO_NON_CONTACT_VERBS,
    is_contact_verb,
)
from src.data.class_splits import HICO_VERBS, COCO_CLASSES, RARE_HOI_IDS, NON_RARE_HOI_IDS


class TestPredicatesAndSplits(unittest.TestCase):
    def test_verb_counts(self):
        self.assertEqual(len(HICO_CONTACT_VERBS), 59)
        self.assertEqual(len(HICO_NON_CONTACT_VERBS), 58)
        self.assertEqual(len(HICO_CONTACT_VERBS) + len(HICO_NON_CONTACT_VERBS), 117)
        self.assertEqual(len(HICO_VERBS), 117)
        self.assertEqual(len(HICO_CONTACT_VERBS & HICO_NON_CONTACT_VERBS), 0)

    def test_contact_predicate_behavior(self):
        self.assertTrue(is_contact_verb("hold"))
        self.assertTrue(is_contact_verb("ride"))
        self.assertTrue(is_contact_verb("sit_on"))
        self.assertTrue(is_contact_verb("carry"))
        self.assertTrue(is_contact_verb("eat"))

        # Non-contact verbs
        self.assertFalse(is_contact_verb("watch"))
        self.assertFalse(is_contact_verb("point"))
        self.assertFalse(is_contact_verb("wave"))
        self.assertFalse(is_contact_verb("no_interaction"))

    def test_class_splits_counts(self):
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(len(RARE_HOI_IDS), 138)
        self.assertEqual(len(NON_RARE_HOI_IDS), 462)
        self.assertEqual(len(RARE_HOI_IDS) + len(NON_RARE_HOI_IDS), 600)
        self.assertEqual(len(set(RARE_HOI_IDS) & set(NON_RARE_HOI_IDS)), 0)


if __name__ == "__main__":
    unittest.main()
