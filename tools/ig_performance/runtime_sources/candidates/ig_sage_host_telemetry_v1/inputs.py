"""Exact original IG graph, native SSD payload, hot sets and 20+300 roots."""
from .common import *


def bind(output,execution):
    from candidates.ig_sage_affinity_pair_v1.inputs import bind as parent_bind
    value=parent_bind(output,PARENT_SHA)
    value.update(candidate_sha256=execution,parent_candidate_sha256=PARENT_SHA,
        optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),
        note='Fresh original GIDS default versus original DiGiT CPU2 ABBA diagnostic. Restore original host stages; external training CPU scheduling and CPU/GPU clocks. Exact original roots/native/cache implementation; inherited native smoke evidence only.')
    write(Path(output)/'inputs.json',value)
    return value
