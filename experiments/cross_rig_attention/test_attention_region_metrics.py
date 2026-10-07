#!/usr/bin/env python3
"""CPU-only checks for the FP/TP attention-region analysis."""

import unittest

import numpy as np

from attention_region_metrics import box_entry_distance, region_labels, score_regions


class RegionMetricTests(unittest.TestCase):
    def test_ray_hits_box_only_in_front_and_in_3d(self):
        origins = np.zeros((4, 3))
        rays = np.asarray([[1., 0., 0.], [-1., 0., 0.],
                           [1., 0., 1.], [0., 1., 0.]])
        box = [5., 0., 0., 2., 2., 2., 0.]
        distances = box_entry_distance(origins, rays, box, 0.0)
        self.assertAlmostEqual(distances[0], 4.)
        self.assertTrue(np.isinf(distances[1:]).all())

    def test_nearest_box_wins_and_overlap_is_recorded(self):
        origins = np.zeros((2, 3))
        rays = np.asarray([[1., 0., 0.], [0., 1., 0.]])
        boxes = [[8., 0., 0., 2., 2., 2., 0.],
                 [4., 0., 0., 2., 2., 2., 0.]]
        labels, overlap = region_labels(origins, rays, boxes,
                                        ['car', 'truck'], 0.0)
        self.assertEqual(labels.tolist(), [1, 2])
        self.assertEqual(overlap.tolist(), [True, False])

    def test_weighted_auc_and_mass(self):
        result = score_regions([.6, .3, .1], [True]*3, [0, 1, 2])
        self.assertAlmostEqual(result['car']['attention_auroc'], 1.0)
        self.assertAlmostEqual(result['car']['attention_mass'], .6)
        self.assertAlmostEqual(result['no_annotated_gt']['attention_auroc'], 0.0)

    def test_auc_uses_average_tie_ranks(self):
        result = score_regions([.2, .2, .6], [True]*3, [0, 1, 2])
        self.assertAlmostEqual(result['car']['attention_auroc'], .25)

    def test_missing_region_has_undefined_auc(self):
        result = score_regions([.7, .3], [True, True], [0, 2])
        self.assertIsNone(result['other_object']['attention_auroc'])
        self.assertEqual(result['other_object']['attention_mass'], 0.0)


if __name__ == '__main__':
    unittest.main()
