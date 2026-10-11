from .layout import materialize
from candidates.ukl_window_native_v2.reference import sample as parent_sample
def sample(a,seeds,f,grouped,seed):return parent_sample(materialize(a),seeds,f,grouped,seed)
