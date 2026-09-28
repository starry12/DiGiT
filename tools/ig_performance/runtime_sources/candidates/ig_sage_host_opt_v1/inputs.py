"""Exact original IG graph, native SSD payload, hot sets and 20+300 roots."""
from .common import *


def bind(output,execution):
    from candidates.ig_sage_stage_profile_v1.inputs import bind as parent_bind
    value=parent_bind(output,PARENT_SHA)
    value.update(candidate_sha256=execution,parent_candidate_sha256=PARENT_SHA,
        optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),
        note='DiGiT-only affinity/frontier-index factorial. GIDS baseline unchanged. Exact original roots and native SSD/cache implementation.')
    write(Path(output)/'inputs.json',value)
    return value
