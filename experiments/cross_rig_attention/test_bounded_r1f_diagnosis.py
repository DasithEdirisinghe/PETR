"""Small dependency-free checks for the bounded 3D diagnostic rules."""

import unittest

from audit_tide3d_car_full import class_ap
from run_bounded_r1f_diagnosis import (ap_with_reduced_gt_count, classify_fp)


def box(x, y, label='car', score=.8):
    return {'translation': [x, y, 0], 'detection_name': label,
            'detection_score': score}


def truth(x, y, label):
    return {'point': (x, y), 'class': label}


class CategoryTests(unittest.TestCase):
    def test_six_prediction_side_categories_and_ambiguity(self):
        token = 'frame'
        car = {token: [truth(0, 0, 'car')]}
        truck = {token: [truth(0, 0, 'truck')]}
        empty_matches = {'car': {}, 'truck': {}}
        self.assertEqual(classify_fp(token, 1, box(1, 0), car,
                         {(token, 0)}, empty_matches, 8)[0], 'Dupe')
        kind, link, _ = classify_fp(token, 1, box(5, 0), car,
                                     set(), empty_matches, 8)
        self.assertEqual((kind, link), ('Loc', (token, 0)))
        self.assertEqual(classify_fp(token, 1, box(20, 0), car,
                         set(), empty_matches, 8)[0], 'Bkgd')
        self.assertEqual(classify_fp(token, 1, box(1, 0), truck,
                         set(), empty_matches, 8)[0], 'Cls')
        self.assertEqual(classify_fp(token, 1, box(5, 0), truck,
                         set(), empty_matches, 8)[0], 'Both')
        crowded = {token: [truth(0, 0, 'car'), truth(1, 0, 'truck')]}
        self.assertEqual(classify_fp(token, 1, box(.5, 0), crowded,
                         set(), empty_matches, 8)[0], 'Ambiguous')

    def test_missed_oracle_changes_denominator_not_ranked_predictions(self):
        token = 'frame'
        gt = {token: [truth(0, 0, 'car'), truth(20, 0, 'car')]}
        preds = {token: [box(0, 0)]}
        baseline = class_ap(preds, gt, 'car')
        corrected = ap_with_reduced_gt_count(preds, gt, 1)
        self.assertGreater(corrected, baseline)
        self.assertAlmostEqual(corrected, 1.0)


if __name__ == '__main__':
    unittest.main()
