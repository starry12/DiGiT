"""Three PA model entries; all previous frozen versions remain unchanged."""
from pathlib import Path
from ae.common import read,write,sha,require
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def verify():
    from artifact_integrity import verify_component
    return verify_component(HERE)
