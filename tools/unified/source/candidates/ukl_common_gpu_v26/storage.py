"""Reuse one pinned successful UKL write; this repair cannot start a writer."""
import json
from pathlib import Path
from candidates.ukl_training_native_v15r11.storage import accepted_storage as original

def accepted_storage():
    path=Path(__file__).resolve().parents[2]/'results/ukl_common_gpu_20261005_v26/ssd_binding.json'
    pinned=json.loads(path.read_text());current=original()
    if current!=pinned:raise RuntimeError('Pinned accepted UKL SSD receipt or feature identity changed')
    return current
