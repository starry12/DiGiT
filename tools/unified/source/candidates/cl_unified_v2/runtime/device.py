from .device_base import Native as BaseNative, Sampler as BaseSampler, S
from candidates.large_graph_fast_v1.sampling import classes
Native,Sampler=classes(BaseNative,BaseSampler,S,'cl')
