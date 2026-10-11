"""Common-loop integration seam for the admitted fresh native worker."""
from functools import partial
from .adapter import IGAdapter
from candidates.unified_training_v2.config import load_config
from candidates.unified_training_v2.loop import run_arm
def run_admitted(*,mode,arm,repetition,output,canonical,bfs_receipt,check,mark,emit,profile_directory):
    from .resources import create
    c=load_config('IG',mode)
    factory=partial(create,output=output,canonical=canonical,bfs_receipt=bfs_receipt,check=check,mark=mark,profile_directory=profile_directory)
    return run_arm(c,arm,repetition,canonical,IGAdapter(factory),emit)
