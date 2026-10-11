"""Exact original IG graph, native SSD payload, hot sets and 20+300 roots."""
from .common import *


def bind(output,execution):
    from candidates.ig_sage_host_opt_v1.inputs import bind as parent_bind
    value=parent_bind(output,PARENT_SHA)
    value.update(candidate_sha256=execution,parent_candidate_sha256=PARENT_SHA,
        optimization_protocol_sha256=sha(HERE/'optimization_protocol.json'),
        note='One matched-root CPU2 pair. Original feature payload and GIDS cache retained; DiGiT uses separately validated v2 adjacency, strict priority and a complete-epoch CPU hot set.')
    from candidates.group_cache_migration_v1.prepare import check_prepared
    value['group_reuse']=check_prepared('IG')
    write(Path(output)/'inputs.json',value)
    return value
