"""Stage CUDA observations; no nvidia-smi polling or external monitor claim."""
from ae.pa_sage.observations import CheckpointObservations as BaseObservations


class CheckpointObservations(BaseObservations):
    def result(self):
        value = super().result()
        value['mode'] = 'cuda_checkpoints_only'
        value['limitation'] = ('CUDA stage checkpoints and PyTorch allocator peaks only; '
                               'no continuous GPU sampling, process-occupancy audit or total-device peak claim.')
        return value
