"""Preview by default; --execute refuses an unfinished grid before any mutation."""
import argparse
import json
from pathlib import Path
from .common import OUT,ROOT,GRID,read,require,write_new,heavy_gate,verify
from .protocol import compile_plan,validate,schedule


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run','cpu-check','verify'),nargs='?',default='run')
    p.add_argument('--protocol',type=Path,default=OUT/'plan/protocol.json')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    require(not a.resume or (a.action=='run' and a.execute),'Resume must be explicit execution')
    if a.action=='run' and a.execute:
        heavy_gate()
        from .controller import run
        run(a.protocol.resolve(),(a.output or OUT/'native').resolve(),a.resume)
    elif a.action=='plan':
        require(not a.execute,'Plan never executes work')
        config=compile_plan();folder=a.output or OUT/'plan';folder.mkdir(parents=True,exist_ok=False)
        write_new(folder/'protocol.json',config);write_new(folder/'schedule.json',schedule(config))
        from .admission import estimate
        manifest=read(Path(config['layout']['base'])/'final/bundle/manifest.json')
        write_new(folder/'budgets.json',{arm:estimate(config,manifest,arm) for arm in config['run_order']})
        write_new(folder/'profile_budget.json',estimate(config,manifest,profile=True))
        print('Plan only; no GPU queries, compilation, large preparation or workers.')
    elif a.action=='cpu-check':
        require(not a.execute,'CPU checks do not execute native work')
        from .tests import run_checks
        run_checks(a.output or OUT/'cpu_checks')
    elif a.action=='verify':
        print(verify())
    else:
        config=validate(read(a.protocol)) if a.protocol.exists() else compile_plan()
        state=read(GRID/'status.json')
        print(json.dumps(dict(execute=False,automatic_start=False,native_accepted=False,
            grid_complete=state.get('complete'),grid_stage=state.get('stage'),
            schedule=schedule(config),large_output=config['execution']['large_output_root'],
            monitor='nvidia-smi',existing_grid_must_finish=True),indent=2))


if __name__=='__main__':main()
