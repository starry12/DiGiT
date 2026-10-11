"""Common frozen-trace accuracy, in a separate process from SSD training."""
import argparse
import hashlib
import math
import time
import numpy as np
from .common import *


def score(model,trace,ids,labels,gather,device,folder,tag,progress=lambda **kw:None):
    import torch
    from candidates.pa_sage_uks_style_v1.labels import classification_targets
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    model.eval();before=model_hash(model);predictions=[];loss_sum=0.;seen=0;calls=0
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.no_grad():
        for inp,out,blocks in trace.iter_batches(device=device):
            current=out.cpu().numpy();expected=ids[seen:seen+len(out)]
            require(np.array_equal(current,expected),'Evaluation root order/coverage changed')
            target=classification_targets(labels,current,cfg()['classes']).to(device)
            logits=model(blocks,gather(inp))
            require(logits.shape==(len(out),cfg()['classes']) and torch.isfinite(logits).all().item(),'Invalid evaluation logits')
            loss_sum+=float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
            predictions.append(logits.argmax(1).cpu().numpy());seen+=len(out);calls+=1
            if calls%25==0:progress(batch=calls,examples=seen)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    require(seen==len(ids) and calls==math.ceil(len(ids)/cfg()['batch_size']),'Incomplete evaluation')
    require(before==model_hash(model),'Evaluation changed model weights')
    pred=np.concatenate(predictions);target=classification_targets(labels,ids,cfg()['classes']).numpy();correct=pred==target
    np.save(folder/(tag+'_predictions.npy'),pred);np.save(folder/(tag+'_nodes.npy'),ids)
    return dict(examples=seen,batches=calls,accuracy=float(correct.mean()),correct=int(correct.sum()),loss=loss_sum/seen,
        seconds=seconds,model_sha256=before,trace_manifest_sha256=trace.manifest_sha256,
        predictions_sha256=sha(folder/(tag+'_predictions.npy')),nodes_sha256=sha(folder/(tag+'_nodes.npy')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['gids','digit'],required=True)
    parser.add_argument('--training',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lock-fds',required=True);a=parser.parse_args()
    heavy_gate();source=verify();validate_inherited_locks(a.lock_fds)
    require(a.output.resolve().parent.parent==OUT/'runs' and a.training.resolve()==a.output.parent/a.arm,'Unexpected evaluation paths')
    a.output.mkdir(exist_ok=False);setup()
    import torch,dgl
    from ae.igb.models import SAGE
    from ae.common import host
    from ae.pa_sage.common import source_config,splits
    from ae.pa_sage.trace_cache import PreparedEvaluationTrace
    from candidates.pa_sage_cache_policy_v1.training import model_hash
    from .dataset import validate_epochs
    report=read(a.training/'report.json');validate_epochs(report)
    require(report['source_sha256']==source and report['arm']==a.arm,'Wrong training source/arm')
    p=cfg();data=source_config();evaluation=read(HERE/'evaluation_inputs.json')
    def progress(stage,**kw):write(a.output/'progress.json',dict(stage=stage,arm=a.arm,updated_unix=time.time(),**kw))
    require(host()>160*2**30,'Evaluation requires 160 GiB available host memory')
    require(torch.cuda.mem_get_info()[0]>4*2**30,'Evaluation requires 4 GiB free GPU memory')
    torch.set_num_threads(1);torch.manual_seed(p['seed']);torch.cuda.manual_seed_all(p['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();progress('prepare_evaluation_traces');traces={};root_ids={}
    for split,binding in evaluation.items():
        root_ids[split]=splits(split);trace=PreparedEvaluationTrace(ROOT/binding['path'],binding['sha256'])
        trace.validate_contract(p['graph']['nodes'],p['graph']['edges'],p['fanouts'],p['batch_size'],root_ids[split])
        trace.prepare(max_bytes=4*2**30);traces[split]=trace
    progress('pin_original_features')
    feature_path=data['source_features']['path'];before=identity(feature_path)
    features=np.load(feature_path,mmap_mode='r');pinned=torch.from_numpy(np.asarray(features)).pin_memory()
    require(pinned.is_pinned() and tuple(pinned.shape)==(p['graph']['nodes'],128),'Invalid raw features')
    digest=hashlib.sha256()
    with open(feature_path,'rb') as f:digest.update(f.read(features.offset))
    view=memoryview(pinned.numpy()).cast('B')
    for lo in range(0,len(view),16*2**20):digest.update(view[lo:lo+16*2**20])
    del view
    require(identity(feature_path)==before and digest.hexdigest()==data['source_features']['sha256'],'Original feature identity/hash changed')
    label_path=data['label_identity']['path'];require(sha(label_path)==data['label_identity']['sha256'],'Labels changed')
    labels=np.load(label_path,mmap_mode='r').reshape(-1)
    net=SAGE(128,p['hidden'],p['classes'],p['layers'],p['dropout']).cuda()
    setup_seconds=time.perf_counter()-started;validation=[]
    gather=lambda inp:dgl.utils.gather_pinned_tensor_rows(pinned,inp)
    for epoch in report['epochs']:
        n=epoch['epoch'];checkpoint=a.training/('epoch%02d.pt'%n)
        require(str(checkpoint)==epoch['checkpoint'] and sha(checkpoint)==epoch['checkpoint_sha256'],'Checkpoint changed')
        net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        require(model_hash(net)==epoch['model_sha256'],'Checkpoint weights changed')
        progress('validation',epoch=n)
        result=score(net,traces['valid'],root_ids['valid'],labels,gather,'cuda:0',a.output,'valid_epoch%02d'%n,
                     lambda **kw:progress('validation',epoch=n,**kw))
        validation.append(dict(epoch=n,**result));write(a.output/'validation.json',validation)
    require(model_hash(net)==report['final_model_sha256'],'Final test must use epoch20')
    progress('final_test',epoch=20)
    test=score(net,traces['test'],root_ids['test'],labels,gather,'cuda:0',a.output,'test_epoch20',
               lambda **kw:progress('final_test',epoch=20,**kw))
    require(verify()==source,'Source changed during evaluation')
    write(a.output/'report.json',dict(passed=True,arm=a.arm,source_sha256=source,validation=validation,test=dict(epoch=20,**test),
        setup_seconds=setup_seconds,validation_calls=20,test_calls=1,optimizer_updates=0,
        training_report_sha256=sha(a.training/'report.json'),training_cache_untouched=True,
        evaluation_feature_path='Verified original features in pinned host memory; same frozen traces for both arms',
        checkpoint_selection='fixed epoch20',raw_ssd_io=False))
    progress('complete',passed=True)

if __name__=='__main__':main()
