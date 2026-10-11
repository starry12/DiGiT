"""All heavy input reads occur after reviewer and SSD exclusion is acquired."""
import subprocess
from .common import *


def prepare(folder,execution):
    from candidates.pa_sage_bidir_native_v2.run import input_binding,identity
    from candidates.pa_sage_bidir_native_v2.common import verify as parent_verify
    binding=input_binding(folder,parent_verify())
    desc=source_config()['splits']['train'];path=Path(desc['path']);before=identity(path)
    require(sha(path)==desc['sha256'] and identity(path)==before,'Training split changed')
    binding['files'][str(path)]=dict(sha256=desc['sha256'],identity=before)
    binding['parent_candidate_sha256']=binding['candidate_sha256']
    binding.update(candidate_sha256=execution,protocol_sha256=sha(P))
    write(folder/'inputs.json',binding)
    with (folder/'bfs_preparation.log').open('x') as log:
        subprocess.run([PY,'-B','-m',MODULE+'.order','--output',str(folder)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    binding=read(folder/'inputs.json')
    require(binding['bfs_preparation']['passed'],'BFS preparation failed')
    return binding
