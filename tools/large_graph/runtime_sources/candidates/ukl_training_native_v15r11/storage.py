"""Accept exactly the audited repair4 storage receipt, never a later run silently."""
import json
from .build import OUT
from candidates.ukl_ssd_writer_v14r4.binding import accepted_storage as current_storage

def accepted_storage():
    value=current_storage();frozen=json.loads((OUT/'ssd_binding.json').read_text())
    for key in ('acceptance_path','acceptance_sha256','source_binding_sha256','arms'):
        if value.get(key)!=frozen[key]:raise RuntimeError('Pinned storage acceptance changed: '+key)
    return value
