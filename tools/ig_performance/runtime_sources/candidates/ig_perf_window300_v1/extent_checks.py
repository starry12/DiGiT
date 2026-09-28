"""Regression checks for extended roots, report rejection and exported rates."""
import copy
import tempfile
import unittest
from unittest import mock
import numpy as np
from .common import *
from .windows import root_slices,check_windows


class Checks(unittest.TestCase):
    def test_320_updates_300_measured_and_no_old_alias(self):
        p=cfg()
        self.assertEqual(root_slices(p),[('warmup',0,20,20),('training',20,320,100)])
        self.assertEqual(root_slices(p,True),[('training',0,2,2)])
        for key,value in [('window_batches',30),('measured_batches',90),
                          ('total_train_batches',110),('epochs',20),
                          ('validation_calls',1),('test_calls',1)]:
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                root_slices(dict(p,**{key:value}))

    def test_prepared_prefix_ranges_and_corruption(self):
        from . import inputs
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);source=d/'source.npy';order=np.random.RandomState(0).permutation(400000).astype('<i8')
            np.save(source,order)
            ready=d/'source_ready.json'
            write(ready,dict(passed=True,protocol_sha256='source-protocol',files={'source.npy':sha(source)}))
            p=dict(cfg(),train_nodes=len(order),train_order=str(d/'prepared/train_order.npy'),
                   source_train_order=str(source),source_order_ready=str(ready),source_order_protocol_sha256='source-protocol')
            with mock.patch.object(inputs,'cfg',return_value=p):
                receipt=inputs.prepare_orders()
                got=np.load(p['train_order'])
                np.testing.assert_array_equal(got,order[:327680])
                self.assertEqual(receipt['warmup_payload_sha256'],digest(order[:20480]))
                self.assertEqual(receipt['smoke_payload_sha256'],digest(order[:2048]))
                self.assertEqual(receipt['training_window_payload_sha256'],[
                    digest(order[20480+j*102400:20480+(j+1)*102400]) for j in range(3)])
                self.assertEqual(receipt,inputs.prepare_orders())
                with Path(p['train_order']).open('r+b') as f:
                    f.seek(-8,2);f.write(b'\x00'*8)
                with self.assertRaisesRegex(RuntimeError,'Prepared order changed'):
                    inputs.prepare_orders()

    def test_stale_110_update_formal_report_rejected(self):
        from .validation import report_check
        from .model import make_model,model_config
        p=cfg()
        r=dict(schema='digit-ig-window-report-v1',passed=True,arm='gids',smoke=False,
            candidate_sha256='c',protocol_sha256='p',model_name='sage',seed=0,epochs=None,
            model_config=model_config('sage'),optimizer=p['optimizer'],
            model_parameter_count=sum(v.numel() for v in make_model('sage').parameters()),
            validation=None,validation_calls=0,test=None,test_calls=0,accuracy=None,
            final_accuracy_claim=False,epoch_time_claim=False,steady_state_proven=False,
            diagnostic_replays=0,raw_ssd_writes=False,all_metadata_and_cache_reused=True,
            route_warmup=False,updates=110,measured_batches=90)
        with self.assertRaisesRegex(RuntimeError,'Incomplete bounded run'):
            report_check(r,'gids',False,dict(candidate_sha256='c',protocol_sha256='p'),Path('/unused'))

    def test_export_uses_300_batch_denominator_and_all_windows(self):
        from . import summarize as summary
        windows=[]
        for j in range(3):
            windows.append(dict(index=j,batch_start=j*100,batch_end=(j+1)*100,batches=100,
                seconds=10.,roots_sha256=str(j),group_edges=0,outer_edges=100,
                device_raw=[0,0,1,1,4096,1000000],device_bytes=4096,primary_bytes=4096,
                replay_bytes=0,useful=dict(ssd_useful_bytes=4096),feature_cpu=1,
                feature_gpu_ssd=1,feature_seconds=.1))
        train=dict(seconds=30.,windows=windows,feature=dict(cpu=3,gpu_ssd=3),
                   group_edges=0,outer_edges=300,roots_sha256='measured-roots',examples=307200)
        warmup=dict(seconds=2.,roots_sha256='warmup-roots',examples=20480,windows=[dict(roots_sha256='warmup')])
        r=dict(model_name='sage',model_config={},optimizer={},initial_parameters_sha256='init',
            seed=0,epochs=None,smoke=False,updates=320,measured_batches=300,training=train,
            warmup=warmup,training_seconds=30.,warmup_seconds=2.,setup_seconds=50.,worker_seconds=82.,
            external_monitor=dict(observed_peak_device_used_bytes=100,strict_monitor_passed=True,timeout_count=0),
            final_parameters_sha256='final')
        io=dict(ssd_completed_bytes=12288,ssd_useful_gbps=1.,ssd_physical_gbps=1.,effective_feature_gbps=1.)
        with mock.patch.object(summary,'phase_metrics',return_value=io):
            v=summary.comparison(dict(gids=r,digit_full=copy.deepcopy(r)),'cpu-fixture',{})
        self.assertEqual(v['rows'][0]['seconds_per_batch'],.1)
        self.assertEqual(v['rows'][0]['training_roots_per_second'],10240.)
        self.assertEqual(v['rows'][0]['updates'],320)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'export';summary.emit(v,folder)
            text=(folder/'README.md').read_text()
            self.assertIn('300 measured batches',text)
            self.assertIn('3 x 100',text)
            self.assertNotIn('90 measured',text)
            self.assertEqual(len((folder/'windows.csv').read_text().splitlines()),7)
        self.assertTrue(check_windows(windows,300,100,30.))
        with self.assertRaises(RuntimeError):check_windows(windows[:2],300,100,20.)


if __name__=='__main__':unittest.main(verbosity=2)
