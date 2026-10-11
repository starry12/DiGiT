"""Reuse the exact accepted 327680-root prefix and verified data identities."""
from .common import *


def bind(output, execution):
    from candidates.ig_perf_window300_v1.inputs import bind as parent_bind
    from candidates.ig_perf_window300_v1.common import P as parent_protocol
    require(sha(P) == sha(parent_protocol), 'Training protocol changed')
    value = parent_bind(output, PARENT_SHA)
    value.update(candidate_sha256=execution, parent_candidate_sha256=PARENT_SHA,
        profile_protocol_sha256=sha(HERE/'profile_protocol.json'),
        note='Exact accepted IG 20+300 roots, graph, cache and native implementation. '
             'Fresh paired 2-batch host-instrumented smoke gates all off/host runs.')
    write(Path(output)/'inputs.json', value)
    return value
