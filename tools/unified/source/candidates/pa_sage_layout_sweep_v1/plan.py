"""Create a 15-point PA layout plan using small metadata files only."""
import argparse
import shutil
from .common import *

def make_plan(output,layout_root=None):
    output=Path(output).resolve()
    layout_root=Path(layout_root).resolve() if layout_root is not None else output.parent/'layouts'
    from ae.pa_sage.common import source_config,csc_paths
    source=source_config();base=ROOT/'data/papers_g2_random_v2'
    prepared=read(base/'prepared.json');native=read(ROOT/'candidates/pa_sage_ablation_v1/protocol.json')
    require(prepared['passed'],'Missing inherited hot-set provenance')
    inputs={}
    for name,path in zip(('indptr','indices'),csc_paths()):
        inputs[name]=dict(path=str(path),sha256=source['source_contract']['csc'][path.name]['sha256'])
    inputs['features']=dict(path=source['source_features']['path'],sha256=source['source_features']['sha256'])
    inputs['hot_nodes']=dict(path=str(base/'hot_nodes.npy'),sha256=prepared['bindings']['hot_nodes.npy'])
    for desc in inputs.values():
        require(Path(desc['path']).is_file(),'Input missing: '+desc['path'])
        desc['observed_identity']=identity(desc['path'])
    storage_probe=existing_parent(layout_root)
    available=shutil.disk_usage(storage_probe).free
    n=source['num_nodes'];e=source['normalized_num_edges'];hot=native['cpu_cache_rows']
    points=[]
    for spec in grid():
        b=budget(n,e,hot,spec['group_size'],spec['replica_percent'])
        points.append(dict(**spec,budget=b,fits_current_free_space_upper_bound=b['filesystem_bytes_with_scratch_and_reserve']<=available,
            filesystem_destination=str(layout_root/spec['id']),
            native_ready=False,raw_ssd_offset=None,
            pending=['filesystem_build','bidirectional_overlay','actual_layout_GPU_admission',
                     'separate_SSD_range_and_readback','parameterized_training_adapter','native_smoke','20_epoch_run']))
    return dict(schema='digit-pa-sage-layout-sweep-plan-v1',created_unix=time.time(),
        scope='code and planning only; no graph array loaded or hashed',
        dataset=dict(nodes=n,directed_edges=e,bidirectional_edges=native['graph']['edges']),
        inputs=inputs,hot_rows=hot,hot_policy='same frozen logical hot-node IDs for every point; rebuild physical CPU row indices',
        grouping_policy='descending normalized DIRECTED degree, stable node ID ties; append reverse edges raw in a separate stage',
        replica_policy='floor(N * percent / 100) additional logical feature rows, rounded down to complete groups; padding is separate',
        seed=0,bfs_enabled=False,training_reference=native,
        training_reference_sha256=sha(ROOT/'candidates/pa_sage_ablation_v1/protocol.json'),
        layout_root=str(layout_root),space_probed_at=str(storage_probe),available_filesystem_bytes_snapshot=available,
        all_points_payload_bytes_upper_bound=sum(p['budget']['payload_bytes_max'] for p in points),
        execution_policy='one new layout at a time after the active ablation; no automatic queue or SSD writes',
        source_bindings_revalidated=False,raw_ssd_access=False,large_preprocessing_started=False,points=points)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layout-root',type=Path,help='Future large layout directory; this command does not create it');a=parser.parse_args()
    require(not a.output.exists(),'Preserve old plan; use a fresh filename')
    require(a.output.parent.is_dir(),'Output parent must exist')
    plan=make_plan(a.output,a.layout_root);write(a.output,plan)
    print(json.dumps(dict(points=len(plan['points']),output=str(a.output),large_preprocessing_started=False),indent=2))

if __name__=='__main__':main()
