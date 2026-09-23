"""Read-only user-space checks; device/kernel readiness is reported separately."""
import importlib.metadata as md
import json, os, platform, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def check():
    observed=json.loads((HERE/'observed.json').read_text()); issues=[];versions={}
    excluded={x.lower().replace('_','-') for x in observed['excluded_local_packages']}
    for pkg in observed['packages']:
        if pkg['name'].lower().replace('_','-') in excluded:continue
        try:value=md.version(pkg['name'])
        except md.PackageNotFoundError:value=None
        versions[pkg['name']]=value
        if value!=pkg['version']:issues.append(pkg['name']+': expected '+pkg['version']+', got '+str(value))
    if platform.python_version()!=observed['python'].split()[0]:issues.append('Python version differs')
    os.environ.setdefault('DGLBACKEND','pytorch')
    imports={}
    for name in ('torch','dgl','numpy','scipy','ogb','pybind11'):
        try:
            mod=__import__(name);imports[name]=str(Path(mod.__file__).resolve())
        except Exception as exc:issues.append(name+' import: '+str(exc))
    cuda_runtime=None
    if 'torch' in sys.modules:
        cuda_runtime=sys.modules['torch'].version.cuda
        if cuda_runtime!=observed['torch_cuda_runtime']:issues.append('Torch CUDA runtime differs')
    pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    if pip.returncode:issues.append('pip check: '+pip.stdout.strip())
    return dict(passed=not issues,issues=issues,python=platform.python_version(),kernel=platform.release(),
        versions=versions,imports=imports,torch_cuda_runtime=cuda_runtime,
        libnvm_device_exists=Path('/dev/libnvm0').exists(),native_device_io_tested=False,
        clean_install_proven=False,scope='Exact user-space package/import validation only')
if __name__=='__main__':
    value=check();print(json.dumps(value,indent=2));sys.exit(0 if value['passed'] else 1)
