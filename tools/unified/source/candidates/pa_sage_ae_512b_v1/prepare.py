"""Bind existing AE files only; no graph/layout or CPU-hot-set generation."""
from .common import *
def prepare(folder,execution):
    from candidates.pa_sage_bidir_native_v2.run import input_binding
    from candidates.pa_sage_bidir_native_v2.common import verify as parent
    require(read(PARENT/'protocol.json')==read(DEPLOYED/'candidates/pa_sage_bidir_native_v2/protocol.json'),'AE protocol differs')
    value=input_binding(folder,parent())
    value['parent_candidate_sha256']=value['candidate_sha256']
    value.update(candidate_sha256=execution,protocol_sha256=sha(P))
    write(folder/'inputs.json',value)
    return value
