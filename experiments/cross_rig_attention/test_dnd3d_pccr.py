#!/usr/bin/env python3
"""Small invariant tests for paired 3D DnD analysis."""

import unittest

from analyze_dnd3d_pccr import candidate_kind, checkpoint, diagnose_model, transition


def truth(name, x, token):
    return {'class': name, 'point': (x, 0.0), 'annotation_token': token}


def box(name, x, score):
    return {'detection_name': name, 'translation': [x, 0.0, 0.0],
            'detection_score': score}


class Dnd3DTests(unittest.TestCase):
    def test_candidate_boundaries(self):
        self.assertEqual(candidate_kind('car', 'truck', 3.99, 8), 'Cls')
        self.assertEqual(candidate_kind('car', 'car', 4.0, 8), 'Loc')
        self.assertEqual(candidate_kind('car', 'truck', 4.0, 8), 'Both')
        self.assertIsNone(candidate_kind('car', 'car', 8.0, 8))

    def test_one_to_one_across_classes(self):
        gt = {'frame': [truth('car', 0.0, 'car_gt'),
                        truth('truck', 2.0, 'truck_gt')]}
        preds = {'frame': [box('bus', 1.7, .9)]}
        states, owners = diagnose_model(gt, preds, {}, 8.0)
        self.assertEqual(states['frame', 1]['status'], 'Cls')
        self.assertEqual(states['frame', 0]['status'], 'Competition')
        self.assertEqual(owners['frame', 0], 'truck_gt')

    def test_official_tp_is_not_reused(self):
        gt = {'frame': [truth('car', 0.0, 'car_gt'),
                        truth('truck', 1.0, 'truck_gt')]}
        preds = {'frame': [box('car', .2, .9)]}
        matches = {'car': {('frame', 0): (0, .2, .9)}, 'truck': {}}
        states, owners = diagnose_model(gt, preds, matches, 8.0)
        self.assertEqual(states['frame', 0]['status'], 'Matched')
        self.assertEqual(states['frame', 1]['status'], 'Competition')
        self.assertEqual(states['frame', 1]['blocked_tp_count'], 1)
        self.assertEqual(owners, {})
        self.assertEqual(transition(states['frame', 0],
                                    {'status': 'Miss'}), 'native_only')

    def test_rank_checkpoint_counts_early_fp(self):
        preds = {'frame': [box('car', 10, .99), box('car', 0, .8)]}
        matches = {'car': {('frame', 1): (0, 0.0, .8)}}
        result = checkpoint(preds, matches, 4, .25)
        self.assertEqual((result['rank'], result['tp'], result['fp']), (2, 1, 1))
        self.assertAlmostEqual(result['precision'], .5)
        self.assertIsNone(checkpoint(preds, matches, 4, .5))


if __name__ == '__main__':
    unittest.main()
