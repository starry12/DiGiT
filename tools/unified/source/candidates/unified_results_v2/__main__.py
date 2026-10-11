import argparse,csv,json
from pathlib import Path
from candidates.unified_training_v2.accounting import FIELDS

def export_csv(path,result):
    columns=['dataset','mode','arm','round','epoch','seconds','training_wall_seconds','warmup_seconds','measured_batches','measured_roots',*FIELDS]
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in result.get('epochs',[]):
            writer.writerow(dict(dataset=result['dataset'],mode=result['mode'],**{k:row.get(k) for k in columns[2:] if k not in FIELDS},**row['counters']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--format',choices=['common','pa-full20'],default='common')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.csv').exists():raise ValueError('Use a new output filename; preserve previous evidence')
    if a.format=='pa-full20':
        from .pa import load
        result=load(a.input)
    else:
        if a.config is None:raise ValueError('--config required for common reports')
        config=json.loads(a.config.read_text());series=json.loads(a.input.read_text())
        if series.get('stage')!='complete' or not series.get('passed') or not series.get('complete'):
            result=dict(schema='digit-unified-results-v2',dataset=config['dataset'],mode=config['execution']['name'],
                passed=False,complete=False,stage=series.get('stage'),error=series.get('error'),speedup=None,
                receipts=series.get('receipts',[]),reason='Incomplete series; no partial median')
        else:
            from .report import summarize
            result=summarize(config,series['receipts'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    export_csv(a.output.with_suffix('.csv'),result)
    print(json.dumps(dict(output=str(a.output),passed=result['passed'],complete=result['complete'],speedup=result.get('speedup'))))

if __name__=='__main__':main()
