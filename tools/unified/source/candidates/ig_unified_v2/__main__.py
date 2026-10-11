import argparse,json
from candidates.unified_training_v2.config import plan
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('short1','short5','full20'),default='short1');a=p.parse_args()
    r=plan('IG',a.mode)
    r.update(native_adapter_status='implemented_not_yet_GPU_accepted',
             entry='python -B -m candidates.ig_unified_v2.start --mode '+a.mode+' --execute',
             native_execution_enabled=True,
             prerequisites=['new_verified_BFS_canonical','new_full_BFS_epoch_profile','fresh_async_GPU_and_feature_routing_checks',
                            'admitted_monitored_worker','normal_exit_and_GPU_release_receipt'])
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
