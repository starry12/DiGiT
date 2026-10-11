"""Dispatch and policy identity regressions; never load the full graph."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
from .tests import fixture
from .report import summarize,validate_worker
from .sampling import graph_sampler
from .common import POLICIES

class Integration(unittest.TestCase):
    def test_both_digit_policies_use_same_v2_overlay(self):
        ordinary=(None,None,None,SimpleNamespace())
        priority=(None,None,None,SimpleNamespace(selection_policy=POLICIES['first']))
        with patch('candidates.uks_group_compare_v2.sampling.graph_sampler',return_value=ordinary) as old, \
             patch('candidates.uks_group_first_v1.sampling.graph_sampler',return_value=priority) as new:
            graph_sampler('digit',0,'weighted');old.assert_called_once_with('digit',0,'v2')
            graph_sampler('digit',0,'first');new.assert_called_once_with('digit',0,'v2')
            graph_sampler('gids',0,'gids');self.assertEqual(new.call_count,1)
        with patch('candidates.uks_group_first_v1.sampling.graph_sampler',return_value=ordinary):
            with self.assertRaises(RuntimeError):graph_sampler('digit',0,'first')

    def test_reject_wrong_policy_or_overlay(self):
        for field,value in [('selection_policy',POLICIES['weighted']),('overlay_variant','v1')]:
            reports=fixture();r=reports['round1_first'];r[field]=value
            with self.assertRaises(RuntimeError):summarize(reports)
            with self.assertRaises(RuntimeError):validate_worker(r,'performance','first','source','prepared',1)

    def test_reports_include_direct_policy_comparison(self):
        reports=fixture()
        for i in range(1,6):reports['round%d_first'%i]['seconds']=5
        summary=summarize(reports)
        self.assertEqual(summary['median_speedup_weighted_over_first'],2)
        self.assertEqual(summary['rounds'][0]['arms']['first']['selection_policy'],POLICIES['first'])

    def test_prior_chain_between_workers_is_still_busy(self):
        from .resources import prior_controller_busy
        for state,busy in [('active',True),('deactivating',True),('inactive',False),('failed',False)]:
            with patch('candidates.uks_group_first_compare_v1.resources.subprocess.check_output',return_value=state):
                self.assertEqual(bool(prior_controller_busy()),busy)
