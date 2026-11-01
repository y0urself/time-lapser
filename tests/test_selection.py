"""Selection contracts for repeated scenes and independent building actions."""

import unittest

import numpy as np

from castlestube_cleanup.selection import DedupeGate, merge_spans, subtract_spans


class ForwardTests(unittest.TestCase):
    def frame(self, value):
        return np.full((8, 8), value, dtype=np.uint8)

    def test_static_run_keeps_only_configured_initial_frames(self):
        gate = DedupeGate(5, 2, "kept")
        self.assertEqual(
            [gate.keep(self.frame(20)) for _ in range(6)],
            [True, True, False, False, False, False],
        )

    def test_kept_reference_preserves_accumulating_changes(self):
        values = [0, 3, 6, 9, 12]
        kept = DedupeGate(5, 1, "kept")
        previous = DedupeGate(5, 1, "previous")
        self.assertEqual(
            [kept.keep(self.frame(v)) for v in values], [True, False, True, False, True]
        )
        self.assertEqual(
            [previous.keep(self.frame(v)) for v in values],
            [True, False, False, False, False],
        )

    def test_context_is_kept_even_after_budget_exhausted(self):
        gate = DedupeGate(5, 1, "kept")
        self.assertEqual(
            [gate.keep(self.frame(0), force=i == 2) for i in range(4)],
            [True, False, True, False],
        )

    def test_threshold_zero_and_exact_boundary_retain_changes(self):
        gate = DedupeGate(0, 1, "kept")
        self.assertTrue(all(gate.keep(self.frame(0)) for _ in range(4)))
        gate = DedupeGate(5, 1, "kept")
        self.assertTrue(gate.keep(self.frame(0)))
        self.assertTrue(gate.keep(self.frame(5)))


class IntervalTests(unittest.TestCase):
    def test_clipping_overlap_and_bridge_preserve_source_coordinates(self):
        spans = [(8, 12), (-2, 3), (2, 5), (6, 7), (10, 15), (3, 3)]
        self.assertEqual(merge_spans(spans, 11), [(0, 5), (6, 7), (8, 11)])
        self.assertEqual(merge_spans(spans, 11, gap=1), [(0, 11)])

    def test_subtraction_handles_cut_spanning_multiple_activity_intervals(self):
        self.assertEqual(
            subtract_spans([(0, 3), (5, 9), (12, 15)], [(2, 13)]), [(0, 2), (13, 15)]
        )
