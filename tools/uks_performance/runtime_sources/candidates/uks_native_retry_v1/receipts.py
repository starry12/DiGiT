"""validate_verify_receipt takes a filesystem path, not parsed JSON."""
from pathlib import Path
from candidates.uks_native_v1.storage import api

def validate_saved_receipt(path,plan):
    return api().validate_verify_receipt(Path(path),plan)
