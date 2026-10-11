import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
manifest = json.loads((here/'source-files.json').read_text())
for name, expected in manifest['files'].items():
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise RuntimeError('Invalid snapshot member')
    path = here/'source'/p
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Snapshot mismatch: '+name)
print('Verified', len(manifest['files']), 'snapshot files; no GPU or SSD access.')
