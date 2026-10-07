#!/usr/bin/env python3
"""No-GPU checks for saved-final-output/query identity association."""

import unittest

from match_saved_final_queries import match_saved_queries


def item(name, score, x, query):
    return {'class_name': name, 'score': score,
            'center_xy': (x, 0.0), 'query_index': query}


class SavedQueryMatchTests(unittest.TestCase):
    def test_saved_order_can_differ(self):
        candidates = [item('car', .9, 1, 5), item('truck', .8, 2, 6),
                      item('car', .7, 3, 7)]
        saved = [dict(class_name='car', score=.7, center_xy=(3, 0)),
                 dict(class_name='car', score=.9, center_xy=(1, 0))]
        self.assertEqual(match_saved_queries(candidates, saved), [7, 5])

    def test_fails_when_score_identity_is_absent(self):
        with self.assertRaisesRegex(ValueError, 'Saved final prediction 0'):
            match_saved_queries([item('car', .9, 1, 5)], [
                dict(class_name='car', score=.8, center_xy=(1, 0))])

    def test_fails_on_ambiguous_duplicate(self):
        with self.assertRaisesRegex(ValueError, '2 unique traced candidates'):
            match_saved_queries([item('car', .9, 1, 5), item('car', .9, 1, 6)], [
                dict(class_name='car', score=.9, center_xy=(1, 0))])

    def test_tolerates_small_rerun_drift_and_audits_it(self):
        queries, audit = match_saved_queries(
            [item('car', .9172306657, -12.269904, 547)],
            [dict(class_name='car', score=.9174131751,
                  center_xy=(-12.2698679, .000575))],
            return_diagnostics=True)
        self.assertEqual(queries, [547])
        self.assertAlmostEqual(audit[0]['score_delta'], .0001825094, places=7)
        self.assertLess(audit[0]['center_delta_m'], .001)

    def test_exact_match_wins_over_second_candidate_inside_drift_window(self):
        candidates = [item('car', .0570387654, 94.390396, 32),
                      item('car', .056, 94.41, 87)]
        saved = [dict(class_name='car', score=.0570387654,
                      center_xy=(94.390396118, 0))]
        self.assertEqual(match_saved_queries(candidates, saved), [32])

    def test_exact_matches_reserved_before_loose_matches(self):
        candidates = [item('car', .90, 1.0, 5),
                      item('car', .91, 1.02, 6)]
        saved = [dict(class_name='car', score=.905, center_xy=(1.01, 0)),
                 dict(class_name='car', score=.90, center_xy=(1.0, 0))]
        self.assertEqual(match_saved_queries(candidates, saved), [6, 5])

    def test_genuine_loose_ambiguity_still_fails(self):
        candidates = [item('car', .90, 1.0, 5),
                      item('car', .91, 1.02, 6)]
        saved = [dict(class_name='car', score=.905, center_xy=(1.01, 0))]
        with self.assertRaisesRegex(ValueError, '2 unique traced candidates'):
            match_saved_queries(candidates, saved)


if __name__ == '__main__':
    unittest.main()
