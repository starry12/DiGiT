"""Reuse accepted native correctness evidence without relabeling its worker version."""
from .common import ROOT,read,sha,require
SOURCE=ROOT/'results/ig_sage_affinity_pair_20260927_v1/native/20260927-100417-3506200/experiment'


def review():
    from candidates.ig_sage_affinity_pair_v1.common import verify,check_inputs
    from candidates.ig_sage_affinity_pair_v1.controller import plan,worker_command,arm_evidence
    from candidates.ig_sage_affinity_pair_v1.review import review_worker
    from candidates.ig_sage_affinity_pair_v1.validation import pair_check
    state=read(SOURCE/'status.json');binding=read(SOURCE/'inputs.json')
    expected='64f861d639361408a5e892ecd5acf57478a3fb5d1b269b770a3d8f46f611f25b'
    require(verify()==state['candidate_sha256']==binding['candidate_sha256']==expected,'Inherited smoke version differs')
    check_inputs(binding);reports={};hashes={}
    for job,worker in zip(plan()[:2],state['workers'][:2]):
        require(all(worker[k]==v for k,v in job.items()) and worker['command'][1:]==worker_command(job,SOURCE)[1:],'Inherited smoke job differs')
        prefix=job['mode']+'_'+job['arm'];receipt=read(SOURCE/(prefix+'_receipt.json'))
        require(receipt['passed'] and receipt['job']==job and receipt['files']==arm_evidence(SOURCE,job),'Inherited smoke evidence differs')
        accepted=SOURCE/(prefix+'_accepted.json')
        require(receipt['accepted_sha256']==sha(accepted) and receipt['report_sha256']==sha(SOURCE/job['mode']/job['arm']/'report.json'),'Inherited smoke hash differs')
        r=review_worker(SOURCE,worker,state['pid'],binding,state['gpu'])
        require(r==read(accepted) and r['external_monitor']['strict_monitor_passed'],'Inherited smoke acceptance differs')
        reports[job['arm']]=r;hashes.update(receipt['files'])
        hashes[prefix+'_accepted.json']=sha(accepted);hashes[prefix+'_receipt.json']=sha(SOURCE/(prefix+'_receipt.json'))
    return dict(passed=True,source=str(SOURCE),candidate_sha256=expected,
                pair=pair_check(reports,True),files=hashes,
                scope='Both accepted native source/routing/edge smokes only; source run incomplete, no formal performance reused.')
