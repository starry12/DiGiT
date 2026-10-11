"""Receipt checks at each controller boundary; no completion from exit code alone."""
import math
from .common import ARMS,read,sha,require,identity,write
from candidates.pa_sage_cache_policy_v2.acceptance import accept_arm,compare,TRACE_KEYS
from candidates.io_accounting_v1.accounting import validate_region


def short(report,p,arm,execution,protocol_sha,prepared_sha,backend_sha):
    require(report['kind']=='native_cache_policy_short' and report['native'] and not report['fixture'] and
            not report['source_only'] and report['passed'] and report['worker_returncode']==0,'Missing native short completion')
    require(report['source_sha256']==execution and report['protocol_sha256']==protocol_sha and
            report['prepared_sha256']==prepared_sha and report['backend_binary_sha256']==backend_sha,'Short input/backend changed')
    require(report['arm']==arm and report['updates']==p['execution']['smoke_updates'] and
            report['evaluation']=='disabled' and report['feature_bit_exact'] and report['sample_edges_verified'] and
            report['finite_loss_and_gradients'],'Short checks did not cover native features/sampling/model')
    require(report['monitor']['passed'] and report['monitor']['backend']=='nvidia-smi','Short monitor failed')
    require(report['cpu_rows']==p['arms'][arm]['cpu_rows'] and
            report['region']['gpu_feature_cache_bytes']==p['arms'][arm]['gpu_feature_cache_bytes'], 'Short budget differs')
    require(report['region']['complete_region'] and report['region']['reconciled'],'Incomplete short counter region')
    validate_region(report['region'])
    served=report['region']['serving']
    require(served['cpu_served_rows']>0 and served['ssd_served_rows']>0,'Short did not exercise CPU and SSD paths')
    require(served['gpu_hit_rows']>0 if arm=='digit' else served['gpu_hit_rows']==0,'Short did not establish bypass/FIFO behavior')
    require(math.isfinite(report['training_seconds']) and report['training_seconds']>0,'Bad short time')
    return report


def make_smoke_matrix(paths,p,execution,protocol_sha,prepared_sha,backend_sha):
    reports={arm:short(read(path),p,arm,execution,protocol_sha,prepared_sha,backend_sha) for arm,path in paths.items()}
    require(set(reports)==set(ARMS),'Every policy needs short acceptance before any full run')
    first=reports['degree']
    for report in reports.values():
        for key in TRACE_KEYS+('exact_smoke_trace_sha256','exact_smoke_features_sha256'):
            require(report[key]==first[key],'Short policies changed workload/features: '+key)
    require(reports['freq']['hot_nodes_sha256']==reports['digit']['hot_nodes_sha256'],'Short hot sets differ')
    return dict(schema='digit-cache-short-matrix-v3',passed=True,source_sha256=execution,
        protocol_sha256=protocol_sha,prepared_sha256=prepared_sha,backend_binary_sha256=backend_sha,
        reports={arm:dict(path=str(path),sha256=sha(path)) for arm,path in paths.items()})


def smoke_matrix(value,p,execution,protocol_sha,prepared_sha):
    require(value['schema']=='digit-cache-short-matrix-v3' and value['passed'] and value['source_sha256']==execution and
            value['protocol_sha256']==protocol_sha and value['prepared_sha256']==prepared_sha,'Wrong short matrix')
    for desc in value['reports'].values():require(sha(desc['path'])==desc['sha256'],'Short acceptance report changed')
    rebuilt=make_smoke_matrix({k:v['path'] for k,v in value['reports'].items()},p,execution,protocol_sha,
                              prepared_sha,value['backend_binary_sha256'])
    require(rebuilt==value,'Short matrix does not match its evidence')
    return value


def full(report,p,arm,execution,protocol_sha,prepared_sha,short_path,backend_sha):
    require(report['fixture'] is False and report['prepared_sha256']==prepared_sha and
            report['backend_binary_sha256']==backend_sha,'Wrong native full inputs')
    accept_arm(report,arm,execution,protocol_sha,sha(short_path),p['arms'][arm])
    require(report['updates']==len(report['shapes']) and
            sum(s['input_nodes'] for s in report['shapes'])==report['region']['serving']['logical_requests'],
            'Full batch/input coverage differs from counters')
    require(report['examples']==p['execution']['training_examples'] and
            sum(s['output_nodes'] for s in report['shapes'])==report['examples'] and
            all(s['input_nodes']>=s['output_nodes']>0 for s in report['shapes']), 'Missing full training examples')
    require(all(s['output_nodes']==p['batch_size'] for s in report['shapes'][:-1]) and
            report['shapes'][-1]['output_nodes']==report['examples']-1178*p['batch_size'], 'Bad batch coverage')
    require(math.isfinite(report['feature_seconds']) and report['feature_seconds']>0, 'Invalid feature time')
    cursor=1
    for window in report['windows']:
        require(window['start_update']==cursor and window['updates']==window['end_update']-cursor+1,'Missing/overlapping windows')
        cursor=window['end_update']+1
    require(cursor==1179+1,'Incomplete full update windows')
    return report


def aggregate(paths,p,execution,protocol_sha,prepared_sha,short_path,backend_sha):
    reports={arm:full(read(path),p,arm,execution,protocol_sha,prepared_sha,short_path,backend_sha) for arm,path in paths.items()}
    result=compare(reports)
    result.update(schema='digit-cache-four-arm-summary-v3',source_sha256=execution,protocol_sha256=protocol_sha,
        prepared_sha256=prepared_sha,native_short_receipt_sha256=sha(short_path),
        reports={arm:dict(path=str(path),sha256=sha(path)) for arm,path in paths.items()})
    for arm,report in reports.items():
        row=result['arms'][arm];dev=report['region']['device'];useful=report['region']['useful_io']
        row.update(order_excluded_seconds=report['order_excluded_seconds'],
            physical_io={k:dev[k] for k in ('primary_bytes','replay_bytes','completed_bytes','primary_commands','replay_commands','active_ns')},
            useful_bytes=useful['ssd_useful_bytes'],feature_seconds=report['feature_seconds'],
            feature_gbps=report['region']['serving']['logical_requests']*512/report['feature_seconds']/1e9,
            speedup_vs_freq=reports['freq']['training_seconds']/report['training_seconds'])
    return result
