"""Bind already accepted AE graph files without rehashing multi-GB inputs live."""
import copy
from .common import *
from candidates.pa_sage_mixed_group_v1.identity import verify_input,filesystem_uuid


def main():
    require(not (HERE/'manifest.json').exists(),'Candidate already frozen')
    predecessor.verify();base=read(reference.OUT/'inputs.json');reference.check_inputs(base)
    previous=ROOT/'results/pa_sage_bfs_decay_20261009_v1/runs/20261009-232414-4187589'
    prior=read(previous/'inputs.json');require(read(previous/'full_summary.json')['passed'],'Unaccepted AE graph predecessor')
    sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
    from stable_identity import Registry
    registry=Registry('ablation')
    binding=copy.deepcopy(base);binding['overlay_files']={};registrations={}
    graph=ROOT/'data/pa_sage_bidir_native_v1';prep=read(graph/'prepared.json')
    for filename in ('original_indptr.npy','original_indices.npy','reordered_indptr.npy','reordered_indices.npy','prepared.json'):
        path=graph/filename;rel=str(path.relative_to(ROOT));old=prior['files'][rel]
        uuid=filesystem_uuid(path)
        rows=registry.by_path.get(str(path),[])
        rows=[r for r in rows if all(r['identity'][k]==old['identity'][k] for k in old['identity'] if k!='device')]
        if filename=='prepared.json':
            require(sha(path)==old['sha256'] and uuid==registrations[str(graph/'original_indices.npy')]['filesystem_uuid'],'Prepared receipt/source filesystem changed')
        else:
            require(rows and uuid in {r['filesystem_uuid'] for r in rows},'Unrecognized prior filesystem: '+str(path))
            registry.verify(path,rows[0]['identity'])
        registration=dict(identity=old['identity'],filesystem_uuid=uuid)
        verify_input(path,old['identity'],registration,identity)
        expected=prior['prepared_sha256'] if filename=='prepared.json' else prep['bindings'][filename]
        require(old['sha256']==expected,'Prior overlay digest differs')
        binding['overlay_files'][str(path)]=old;registrations[str(path)]=registration
    require(sha(graph/'prepared.json')==prior['prepared_sha256'],'Prepared receipt changed')
    binding.update(csc={k:str(graph/('original_'+k+'.npy')) for k in ('indptr','indices')},
        overlay={k:str(graph/(k+'.npy')) for k in ('reordered_indptr','reordered_indices')},
        prepared=str(graph/'prepared.json'),prepared_sha256=sha(graph/'prepared.json'),
        graph_assumption='AE bidirectional CSC; int64 CSC-position EIDs; anonymous pinned copies',
        parent_inputs_sha256=sha(predecessor.OUT/'inputs.json'),overlay_prior_inputs=str(previous/'inputs.json'))
    write(OUT/'inputs.json',binding);write(HERE/'identity_registry.json',dict(files=registrations))
    check_inputs(binding);print('AE graph binding verified by UUID, inode, size, mtime and ctime')


if __name__=='__main__':main()
