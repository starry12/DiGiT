"""Metadata-only plan/bind-inputs; intentionally has no execute/write command."""
import argparse
import json
from pathlib import Path
from . import plan as P


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','bind-inputs'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing evidence: '+str(args.output))
    snapshot=P.inventory()
    if args.action=='plan':
        if args.plan is not None:raise ValueError('--plan only applies to bind-inputs')
        result=P.make_plan(snapshot)
        args.output.mkdir();save(args.output/'inventory.json',snapshot);save(args.output/'plan.json',result)
    else:
        if args.plan is None:raise ValueError('--plan required')
        result,_=P.read_metadata(args.plan);P.validate_plan(result,snapshot)
        from candidates.ukl_training_adapter_v12.inputs import accepted_inputs
        accepted=accepted_inputs()
        binding=P.bind_sources(result,snapshot,accepted)
        # Recheck identities immediately before publishing metadata; no source reads.
        for arm in binding['arms'].values():
            s=arm['source'];path=Path(s['path'])
            if path.resolve(strict=True)!=path or P.ident(path.stat())!=s['identity']:
                raise RuntimeError('Accepted source changed')
        P.validate_plan(result,P.inventory())
        args.output.mkdir();save(args.output/'source_binding.json',binding)
    print(json.dumps(dict(output=str(args.output),action=args.action,raw_device_opened=False,
                          raw_ssd_written=False,registry_reserved=False,raw_ssd_bound=False)))


if __name__=='__main__':main()
