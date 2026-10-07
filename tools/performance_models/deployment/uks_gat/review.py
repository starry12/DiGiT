"""Rebuild UKS acceptance from immutable per-worker evidence, without CUDA."""
import model_context
import hashlib,json,math
from pathlib import Path

SOURCE='e5895a2cccadb62c5bc349aa6e5a403ceba46a7159a4e80e53ba0aa94496b8d1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)

def review(root,expected_gpu=None,transport_sha=None):
    root=Path(root);state=read(root/'status.json');summary=read(root/'summary.json')
    require(state['complete'] and state['passed'] and state['source_sha256']==SOURCE,'Incomplete controller')
    if expected_gpu is not None:
        require(read(root/'gpu_assignment.json')['selected']==expected_gpu and summary.get('gpu_assignment')==expected_gpu and state.get('gpu_assignment')==expected_gpu,'Controller GPU identity changed')
    reports={}
    for name in ['smoke_gids','smoke_digit']+['round%d_%s'%(i,a) for i in range(1,6) for a in ('gids','digit')]:
        p=root/name;r=read(p/'report.json');model_context.checked(r);a=read(p/'accepted.json')
        require(read(p/'exit.json')['returncode']==0 and a['normal_exit'] and r['passed'],'Worker failed: '+name)
        require(a==dict(r,normal_exit=True,report_sha256=sha(p/'report.json')),'Acceptance drift: '+name)
        require(r['source_sha256']==SOURCE and r['cache_line_bytes']==1024 and not r['raw_ssd_writes'],'Protocol drift')
        if expected_gpu is not None:
            require(r.get('gpu_assignment')==expected_gpu and r.get('ae_transport_sha256')==transport_sha,'GPU or transport changed: '+name)
        if name.startswith('round'):
            require(r['updates']==320 and r['measured_batches']==300 and r['warmup_batches']==20,'Wrong window')
            require(r['training']['reconciled'] and math.isfinite(r['seconds']) and r['seconds']>0,'Invalid measurements')
            require(r['hot_policy']==('revpr' if r['arm']=='gids' else 'freq100_seed23'),'Hot policy drift')
            require(r['bfs_enabled']==(r['arm']=='digit'),'Order policy drift')
        reports[name]=r
    measured=[r for n,r in reports.items() if n.startswith('round')]
    require(len({r['initial_model_sha256'] for r in measured})==1,'Model initialization differs')
    for arm in ('gids','digit'):
        own=[r for r in measured if r['arm']==arm]
        for k in ('roots_sha256','hot_file_sha256'):require(len({r[k] for r in own})==1,'Within-arm inputs changed')
    ratios=[reports['round%d_gids'%i]['seconds']/reports['round%d_digit'%i]['seconds'] for i in range(1,6)]
    require(summary['passed'] and summary['node_sets_differ'] and summary['source_sha256']==SOURCE,'Summary context drift')
    require(summary['paired_speedups']==ratios and summary['max_paired_speedup']==max(ratios),'Summary numerical drift')
    return dict(model_identity=model_context.verify(),passed=True,gpu_assignment=expected_gpu,ae_transport_sha256=transport_sha,speedup=max(ratios),selection='maximum_same_round_speedup_out_of_five',
        node_sets_differ=True,workers=10,source_sha256=SOURCE,summary_sha256=sha(root/'summary.json'),
        evidence={str(p.relative_to(root)):sha(p) for d in reports for p in [root/d/n for n in ('report.json','accepted.json','exit.json')]})
