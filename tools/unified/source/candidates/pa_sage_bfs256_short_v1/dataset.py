import numpy as np
from .common import *
from candidates.pa_sage_uks_style_v1.dataset import graph_sampler,roots as bfs_roots
from candidates.cache_epoch_v1.core import digest_array


def random_order(training,seed=0,epoch=0):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(training)


from candidates.pa_sage_bfs256_full20_v1.dataset import roots


def check_report_roots(report,arm):
    complete=roots(arm=arm);n=(24 if report['smoke'] else 320)*1024
    require(report['roots_sha256']==digest_array(complete[:n]),'Actual roots differ from arm policy')
    require(report['full_order_sha256']==digest_array(complete),'Wrong full order provenance')
    require(report['root_order_policy']==cfg()['root_order_by_arm'][arm],'Wrong order description')


def comparison_roots():
    a=roots(arm='gids');b=roots(arm='digit')
    require(np.array_equal(np.sort(a),np.sort(b)),'Full training sets differ')
    aw=a[20*1024:320*1024];bw=b[20*1024:320*1024]
    return dict(full_training_set_equal=True,matched_root_order=False,matched_short_window_examples=False,
                measured_roots_each=len(aw),measured_overlap_nodes=len(np.intersect1d(aw,bw,assume_unique=True)),
                gids_full_order_sha256=digest_array(a),digit_full_order_sha256=digest_array(b),
                gids_measured_order_sha256=digest_array(aw),digit_measured_order_sha256=digest_array(bw))
