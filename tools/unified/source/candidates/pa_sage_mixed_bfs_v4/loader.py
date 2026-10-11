"""Keep one native async owner/cache across epochs; reset only drained Python state."""
from candidates.pa_sage_mixed_group_v1.loader import PairOverlapGIDS
from candidates.pa_sage_512b_pair_v4.loader import BoundedGIDS


class GIDSEpochLoader(BoundedGIDS):
    """Original synchronous baseline with an explicit drained-epoch guard."""
    def begin_epoch(self):
        previous=getattr(self,'pipeline',None)
        if previous is not None and not previous.summary()['drained']:
            raise RuntimeError('Previous GIDS epoch is not drained')
        super().begin_epoch()

    def drain(self):
        # Each baseline read is synchronous; no async owner or pointer ticket.
        pass


class MultiEpochGIDS(PairOverlapGIDS):
    def begin_epoch(self):
        if not hasattr(self,'io_stream'):
            return super().begin_epoch()
        if self.live_ticket is not None or self.hint_refs:
            raise RuntimeError('Previous epoch still owns read pointers')
        if self.pipeline is None or not self.pipeline.summary()['drained']:
            raise RuntimeError('Previous epoch was not drained')
        self.drain()
        # The native owner refuses a second prepare_async_reads. Preserve its
        # stream and the cache; previous counters are already in the epoch report.
        BoundedGIDS.begin_epoch(self)
        self.submitted_reads=self.collected_reads=0
        self.prepare_host_ns=self.collect_host_ns=0
