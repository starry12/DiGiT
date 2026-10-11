"""Bridge the guarded UKL session to the common persistent training loop."""
import os
from types import SimpleNamespace
import numpy as np
from .common import config,verify,write
from .adapter import CLAdapter
from candidates.unified_training_v2.loop import run_arm
from candidates.unified_training_v2.order import roots_digest
from candidates.unified_training_v2.config import digest

def prepare_roots(accepted,nodes,check,event):
    from candidates.group_cache_migration_v1.io import read_array
    from candidates.group_cache_migration_v1.core import validate_order
    files=accepted['stages']['graph']['files']
    event('full_root_validation')
    canonical=read_array(files['bfs.i64'],check=check);training=read_array(files['train.i64'],check=check)
    validate_order(canonical,training,nodes);del training
    event('full_root_validation_complete')
    canonical.flags.writeable=False
    return canonical

def train(arm,native,sampling,model,opt,provider,accepted,output,check,event,mode,repetition,canonical=None):
    from candidates.ukl_postprocess_v22.request import request_from_blocks
    from .runtime.counters import interval
    from .runtime.timing import TIMING
    files=accepted['stages']['graph']['files']
    if canonical is None:canonical=prepare_roots(accepted,native.graph.nodes,check,event)
    _,Sampler=sampling.select(arm)
    source_digest=verify()
    event('training_source_verified',manifest_sha256=source_digest)
    receipt=dict(source_sha256=source_digest,graph_sha256=digest(accepted['stages']['graph']),split_sha256=files['train.i64']['sha256'],
        canonical_sha256=roots_digest([canonical]),cpu_feature_bytes=provider.cache['cpu_feature_bytes'],
        gpu_feature_bytes=provider.cache['gpu_feature_bytes'],worker_id=str(output),evidence_scope='native',
        roots_verified_exact=True,gpu_resident_rows_start=provider.snapshot()['gpu']['resident_pages'])
    def training_check():
        check();verify()
    resources=SimpleNamespace(model=model,optimizer=opt,sampler=Sampler(native,grouped=arm=='digit',seed=0),provider=provider,
        native=native,request=request_from_blocks,snapshot=provider.snapshot,interval=interval,receipt=receipt,output=output,check=training_check,event=event)
    adapter=CLAdapter(resources)
    def emit(row):
        write(output/'common_progress.json',row)
        if row['stage']=='epoch_complete':write(output/('epoch_%02d.json'%row['epoch']),row)
        event('common_'+row['stage'],epoch=row['epoch'],updates=adapter.updates)
    event('training_loop_begin')
    report=run_arm(config(mode),arm,repetition,canonical,adapter,emit)
    write(output/'common_training.json',report)
    measured=[x for x in adapter.window_records if x['phase']=='training']
    def group_stats(windows):
        g=sum(x['groups'] for x in windows);e=sum(x['edges'] for x in windows)
        return dict(grouped_sampled_edges=g,outer_sampled_edges=e,grouped_edge_fraction=g/e if e else 0)
    result=dict(passed=True,updates=report['updates'],warmup_batches=config(mode)['execution']['warmup_batches'],
        measured_batches=sum(e['measured_batches'] for e in report['epochs']),seconds=report['measured_seconds'],
        warmup_seconds=sum(e['warmup_seconds'] for e in report['epochs']),initial_model_sha256=adapter.initial,
        final_model_sha256=report['final']['final_model_sha256'],finite=report['final']['finite'],
        counters=dict(whole=adapter.whole,measured=adapter.measured),accuracy_evaluated=False,
        group_sampling=dict(whole=group_stats(adapter.window_records),measured=group_stats(measured)),
        timing_protocol=dict(TIMING),common_training=report,
        execution_capabilities=config(mode)['execution_capabilities'])
    return result,adapter.feature_checks
