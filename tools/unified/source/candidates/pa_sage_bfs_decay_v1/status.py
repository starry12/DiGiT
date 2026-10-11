"""Read progress or verified final results without starting experiments."""
from .common import *


def main():
    latest=OUT/'latest.json'
    if not latest.exists():
        print('NOT_STARTED');return
    value=read(latest);folder=Path(value['output'])
    require(folder.resolve().parent==(OUT/'runs').resolve(),'Unexpected output path')
    print('State:',value['stage'])
    print('Output:',folder)
    if value.get('error'):print('Error:',value['error'])
    worker=value.get('worker',{})
    if worker.get('status')=='running':
        path=folder/worker['mode']/worker['arm']/'progress.json'
        if path.exists():print('Progress:',json.dumps(read(path),ensure_ascii=False))
    if value.get('complete') and value.get('passed'):
        require(sha(folder/'full_summary.json')==value['summary_sha256'],'Summary changed')
        summary=read(folder/'full_summary.json')
        require(summary['passed'] and summary['strict_resource_acceptance'],'Incomplete acceptance')
        for arm,digest in summary['report_sha256'].items():
            require(sha(folder/('full_'+arm+'_accepted.json'))==digest,'Accepted report changed')
        for arm,row in summary['rows'].items():
            print('%s: mean epoch %.2f s; epoch20 validation %.4f%%; final test %.4f%%' % (
                arm,row['mean_training_epoch_seconds'],100*row['epoch20_validation_accuracy'],100*row['test_accuracy']))
        for arm,row in summary['rows'].items():
            print('%s vs random: %.4fx; test accuracy delta: %+.4f percentage points' % (
                arm,row['speedup_vs_random'],row['test_accuracy_delta_percentage_points']))


if __name__=='__main__':main()
