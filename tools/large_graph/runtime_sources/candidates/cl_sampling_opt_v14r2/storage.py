"""Reuse one pinned successful CL write; this repair cannot start a writer."""
import json
from pathlib import Path
from candidates.cl_ssd_writer_v10.binding import accepted_storage as original

def accepted_storage():
    path=Path(__file__).resolve().parents[2]/'results/cl_sampling_opt_20261005_v14r2/ssd_binding.json'
    pinned=json.loads(path.read_text());current=original()
    if current!=pinned:raise RuntimeError('Pinned accepted CL SSD receipt or feature identity changed')
    return current
