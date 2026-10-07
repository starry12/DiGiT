"""Build eight independent services from the currently installed, repaired controls."""
import ast
import hashlib
import json
import shutil
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
ADMIN = Path('/srv/digit-ae/admin')
DEST = ADMIN / 'multimodel_v2'
SOURCES = dict(IG='ig_performance_v1', UKS='uks_performance_v2', UKL='ukl_performance_v2', CL='cl_performance_v1')
P = H / 'payload'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def replace(source, old, new):
    if old not in source:
        raise RuntimeError('Missing source anchor: ' + old)
    return source.replace(old, new)


def inject(source, snippet):
    """Insert after the module docstring, preserving future imports if any."""
    tree = ast.parse(source)
    pos = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pos = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == '__future__':
            pos = node.end_lineno
        else:
            break
    lines = source.splitlines(True)
    lines.insert(pos, snippet + '\n')
    return ''.join(lines)


def main():
    if P.exists():
        shutil.rmtree(P)
    runtime = P / 'runtime'
    for f in (ROOT / 'candidates/multimodel_performance_v1').glob('*.py'):
        write(runtime / f.name, f.read_text())
    files = {str(p.relative_to(runtime)): sha(p) for p in sorted(runtime.glob('*.py'))}
    write(P / 'runtime_manifest.json', dict(files=files))
    adapter_sha = sha(P / 'runtime_manifest.json')
    configs, source_files = [], {}
    for ds, parent in SOURCES.items():
        source = ADMIN / parent
        for model in ('gcn', 'gat'):
            key = ds.lower() + '_' + model
            control = P / key
            installed = DEST / key
            unit = f'digit-ae-{ds.lower()}-{model}-performance.service'
            selftest = unit.replace('.service', '-selftest.service')
            output = Path('/srv/digit-ae') / f'{ds.lower()}-{model}-performance-results'
            # Retain all immutable parent snapshot/data identities. Model changes
            # are accounted for separately by runtime_manifest and model_context.
            for name in ('snapshot_manifest.json', 'snapshot_identity.json'):
                source_files[str(source / name)] = sha(source / name)
                write(control / name, (source / name).read_text())
            old_unit = next(p for p in (source / 'units').glob('*sage*service'))
            source_files[str(old_unit)] = sha(old_unit)
            unit_text = old_unit.read_text()
            old_output = {'IG': 'ig-performance-results', 'UKS': 'uks-performance-results',
                          'UKL': 'ukl-performance-results-v2', 'CL': 'cl-performance-results'}[ds]
            def remap(s):
                return s.replace(str(source), str(installed)).replace('/srv/digit-ae/' + old_output, str(output)).replace(old_unit.name, unit)
            for f in source.glob('*.py'):
                if f.name in ('legacy_cli.py', 'mount_data.py'):
                    continue
                source_files[str(f)] = sha(f)
                write(control / f.name, remap(f.read_text()))
            context = f'''"""Fixed {ds}/{model} protocol; isolated service, no user-selected Python code."""
import hashlib,json,sys
from pathlib import Path
BASE=Path({str(DEST)!r})
sys.path.insert(0,str(BASE))
sys.path.insert(0,'/home/embed/digit')
from runtime.models import identity,require_identity,cpu_check
DATASET={ds!r}
MODEL={model!r}
EXPECTED_RUNTIME_SHA={adapter_sha!r}
EXPECTED=identity(DATASET,MODEL,EXPECTED_RUNTIME_SHA)
def verify():
 manifest=BASE/'runtime_manifest.json'
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=EXPECTED_RUNTIME_SHA:raise RuntimeError('Model adapter manifest changed')
 for name,digest in json.loads(manifest.read_text())['files'].items():
  p=BASE/'runtime'/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model adapter changed: '+name)
 control=Path(__file__).resolve().parent
 transport=control/'transport_manifest.json'
 for name,digest in json.loads(transport.read_text())['files'].items():
  p=control/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model service changed: '+name)
 EXPECTED['service_sha256']=hashlib.sha256(transport.read_bytes()).hexdigest()
 return EXPECTED
def activate():
 verify()
 # Install UUID-compatible identity hooks before importing any parent module
 # that captures an identity function with a from-import.
 sys.path.insert(0,'/srv/digit-ae/admin/identity_v1')
 import stable_identity
 stable_identity.install(DATASET)
 from runtime.adapter import activate as apply
 apply(DATASET,MODEL,EXPECTED)
def checked(report):
 require_identity(report,verify())
 if DATASET in ('IG','UKL','CL'):
  from runtime.gates import validate as gate_valid
  gates=[row.get('warmup_gate') for row in report['rows']] if DATASET=='IG' and 'rows' in report else [report.get('warmup_gate')]
  if not gates or not all(gate_valid(g) for g in gates):raise RuntimeError('Missing new-model native warmup gate')
 if DATASET in ('UKL','CL') and MODEL=='gat':
  from runtime.probe import validate
  if not validate(report.get('model_memory_probe',{{}}),DATASET,MODEL):raise RuntimeError('Missing bounded GAT memory probe')
 return report
def selftest():
 result=cpu_check(DATASET,MODEL)
 print(json.dumps(dict(result,model_identity=verify())))
 return result
'''
            write(control / 'model_context.py', context)
            preamble = "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nimport model_context\nmodel_context.activate()"
            # Apply in fresh worker processes and import checks. Controllers that
            # rebuild parent acceptance need the same stricter model validator.
            for name in ('runner.py', 'worker.py', 'check_imports.py'):
                f = control / name
                if f.exists():
                    s = inject(f.read_text(), preamble)
                    if name == 'worker.py' and ds in ('UKL', 'CL'):
                        # Framework imports must be charged to the admitted init
                        # cgroup, after the original phase handshake.
                        s = s.replace('model_context.activate()\n', '')
                        s = replace(s, "  await_phase(out,'init')", "  await_phase(out,'init')\n  model_context.activate()")
                    if name == 'check_imports.py':
                        s += '\nmodel_context.selftest()\n'
                    write(f, s)
            f = control / 'cli.py'
            s = f.read_text().replace("choices=('sage',)", f"choices=('{model}',)")
            s = s.replace(' sage', ' ' + model).replace('/SAGE', '/' + model.upper()).replace('/ SAGE', '/ ' + model.upper())
            # New models have no reference measurements; never reuse SAGE receipts.
            if "if a.reference and a.action!='results'" in s:
                s = s.replace("if a.reference and a.action!='results'", "if a.reference")
            if ds == 'IG':
                s = replace(s, "final=read(out/'result.json');review=read(out/'completion_review.json')",
                            "final=read(out/'result.json');review=read(out/'completion_review.json');model_context.checked(review)")
            s = inject(s, "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nimport model_context")
            write(f, s)
            if ds in ('UKL', 'CL'):
                f = control / 'review.py'
                s = inject(f.read_text(), 'import model_context')
                s = replace(s, "a=read(p);w=a['worker'];s=a['worker_state']", "a=read(p);w=a['worker'];s=a['worker_state'];model_context.checked(w)")
                s = replace(s, 'return dict(passed=True,formal_workers=10', 'return dict(model_identity=model_context.verify(),passed=True,formal_workers=10')
                write(f, s)
                f = control / 'common.py'
                s = inject(f.read_text(), 'import model_context')
                s = replace(s, 'return dict(snapshot_sha256=', 'return dict(model_identity=model_context.verify(),snapshot_sha256=')
                write(f, s)
            elif ds == 'UKS':
                f = control / 'worker.py'
                write(f, replace(f.read_text(), "dict(result,source_sha256=verify()", "dict(result,model_identity=model_context.verify(),source_sha256=verify()"))
                f = control / 'controller.py'
                s = inject(f.read_text(), 'import model_context')
                # Validate both systems before starting five measured pairs.
                s = replace(s, "JOBS=[('smoke_digit','digit','smoke')]", "JOBS=[('smoke_gids','gids','smoke'),('smoke_digit','digit','smoke')]")
                s = replace(s, "report=read(folder/'report.json');require", "report=read(folder/'report.json');model_context.checked(report);require")
                write(f, s)
                f = control / 'review.py'
                s = inject(f.read_text(), 'import model_context')
                s = replace(s, "['smoke_digit']+", "['smoke_gids','smoke_digit']+")
                s = replace(s, "r=read(p/'report.json');a=read(p/'accepted.json')", "r=read(p/'report.json');model_context.checked(r);a=read(p/'accepted.json')")
                s = replace(s, 'return dict(passed=True,gpu_assignment=', 'return dict(model_identity=model_context.verify(),passed=True,gpu_assignment=')
                write(f, s)
                f = control / 'runner.py'
                write(f, replace(f.read_text(), 'return dict(snapshot_sha256=', 'return dict(model_identity=model_context.verify(),snapshot_sha256='))
            else:
                # Copy the repaired device-routing control; global GPU reservation
                # and telemetry helpers remain the installed trusted implementation.
                for name in ('ig_controller.py', 'ig_worker.py'):
                    src = ADMIN / 'gpu_selection_v1' / name
                    source_files[str(src)] = sha(src)
                    s = src.read_text()
                    s = s.replace("model='sage'", f"model='{model}'").replace("'--model', 'sage'", f"'--model', '{model}'").replace('IG/SAGE', 'IG/' + model.upper())
                    s = s.replace('/srv/digit-ae/admin/gpu_selection_v1/ig_worker.py', str(installed / 'ig_worker.py'))
                    if name == 'ig_worker.py':
                        s = replace(s, 'from candidates.ig_sage_host_telemetry_v1 import telemetry,worker', 'from candidates.ig_sage_host_telemetry_v1 import telemetry\n    import ig_native_worker as worker')
                        s = replace(s, '    worker.main()', '    sys.path.insert(0,str(Path(__file__).resolve().parent))\n    worker.main()')
                    else:
                        s = replace(s, 'return dict(gpu_assignment=auto_gpu.assignment(),transport_sha256=auto_gpu.transport(),passed=True', 'return dict(model_identity=model_context.verify(),gpu_assignment=auto_gpu.assignment(),transport_sha256=auto_gpu.transport(),passed=True')
                        s = replace(s, "return dict(mode=job['mode'], arm=job['arm'], repetition=job['repetition'],", "return dict(warmup_gate=report['warmup_gate'],mode=job['mode'], arm=job['arm'], repetition=job['repetition'],")
                    write(control / name, inject(s, preamble))
                source_worker = ROOT / 'candidates/ig_sage_host_telemetry_v1/worker.py'
                source_files[str(source_worker)] = sha(source_worker)
                s = source_worker.read_text().replace("choices=['sage']", f"choices=['{model}']")
                s = replace(s, "    require(os.geteuid()==0 and __debug__", "    require(not a.smoke,'New models use the four-update warmup gate in each performance worker')\n    require(os.geteuid()==0 and __debug__")
                s = replace(s, "report=dict(schema='digit-ig-window-report-v1'", "report=dict(model_identity=model_context.verify(),schema='digit-ig-window-report-v1'")
                s = replace(s, "raw=np.memmap(source['feature']['path'],dtype='<f4',mode='r',shape=(p['nodes'],1024)) if a.smoke else None",
                            "raw=np.memmap(source['feature']['path'],dtype='<f4',mode='r',shape=(p['nodes'],1024))")
                s = replace(s, "    original_ptrs=[x.data_ptr()", "    from runtime.gates import WarmupGate\n    gate=WarmupGate(model,opt)\n    model_feature_checks=[]\n    original_ptrs=[x.data_ptr()")
                s = replace(s, "        retained();model.train();features.store.begin_useful_io_region()",
                            "        if name=='training' and not a.smoke:gate.require()\n        retained();model.train();features.store.begin_useful_io_region()")
                s = replace(s, "            opt.zero_grad(set_to_none=True);loss.backward();opt.step()",
                            "            opt.zero_grad(set_to_none=True);loss.backward();opt.step()\n"
                            "            if name=='warmup' and i<4:\n"
                            "                selected=np.linspace(0,len(inputs)-1,min(32,len(inputs)),dtype=np.int64)\n"
                            "                logical=inputs[torch.as_tensor(selected,device=inputs.device)].cpu().numpy()\n"
                            "                actual=x[torch.as_tensor(selected,device=x.device)].detach().cpu().numpy()\n"
                            "                require(np.array_equal(actual,np.asarray(raw[logical])),'Model warmup feature values differ')\n"
                            "                model_feature_checks.append(dict(passed=True,rows=len(selected),batch=i))\n"
                            "                gate.observe(loss,i+1,len(model_feature_checks))")
                s = replace(s, "report=dict(model_identity=model_context.verify(),schema=", "report=dict(warmup_gate=gate.require(),model_feature_checks=model_feature_checks,model_identity=model_context.verify(),schema=")
                write(control / 'ig_native_worker.py', inject(s, preamble))
                f = control / 'runner.py'
                s = replace(f.read_text(), '    import ig_controller as c', '    sys.path.insert(0,str(CONTROL))\n    import ig_controller as c')
                s = replace(s, 'return dict(snapshot_sha256=', 'return dict(model_identity=model_context.verify(),snapshot_sha256=')
                write(f, s.replace('# IG/SAGE', '# IG/' + model.upper()))
            unit_text = remap(unit_text).replace(' SAGE ', ' ' + model.upper() + ' ')
            # All adapter code is in /srv, visible inside the existing private namespace.
            write(control / 'units' / unit, unit_text)
            self_text = unit_text.replace('Type=exec', 'Type=oneshot').replace('/runner.py\n', '/runner.py --selftest\n')
            self_text = '\n'.join(line for line in self_text.splitlines() if not line.startswith('RuntimeMaxSec=')) + '\n'
            write(control / 'units' / selftest, self_text)
            write(control / 'transport_manifest.json', dict(files={str(p.relative_to(control)): sha(p) for p in sorted(control.rglob('*')) if p.is_file() and p.name != 'transport_manifest.json'}))
            configs.append(dict(dataset=ds, model=model, key=key, control=str(installed), output=str(output), unit=unit, selftest=selftest))
    write(P / 'config.json', dict(services=configs, adapter_sha256=adapter_sha))
    write(H / 'parent_controls.json', source_files)
    sudo = '\n'.join('atc27_ae ALL=(root) NOPASSWD: /usr/bin/systemctl --no-block ' + action + ' ' + c['unit'] for c in configs for action in ('start', 'stop')) + '\n'
    write(H / 'sudoers', sudo)
    for p in P.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    print('Prepared', len(configs), 'model services; runtime', adapter_sha)


if __name__ == '__main__':
    main()
