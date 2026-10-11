"""Export accepted IG pairs using the shared byte-aware result schema."""
import json
from pathlib import Path
from candidates.unified_results_v2.report import summarize
from candidates.unified_results_v2.__main__ import export_csv

def export(output,config,series):
    if config['dataset']!='IG':raise ValueError('IG report only')
    if series.get('stage')!='complete' or not series.get('passed') or not series.get('complete'):
        raise ValueError('Do not summarize incomplete IG pairs')
    result=summarize(config,series['receipts'])
    output=Path(output)
    if any((output/name).exists() for name in ('summary.json','summary.csv')):
        raise ValueError('Preserve existing summary')
    with (output/'summary.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    export_csv(output/'summary.csv',result)
    return result
