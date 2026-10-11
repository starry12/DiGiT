"""Refresh after reviewed source changes; this does not recertify old results."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
files={}
for base in ('ae','configs'):
 for path in (ROOT/base).rglob('*'):
  if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and path.name!='release_manifest.json':
   files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in list((ROOT/'data/papers/ssd').glob('*.json'))+[ROOT/'bam/build/lib/libnvm.so',ROOT/'data/igb/full/metadata_ready.json',ROOT/'digit_paths.py',ROOT/'data/papers/bundle/manifest.json',ROOT/'data/igb/full/manifest.json',ROOT/'data/igb/csc/manifest.json']:
 files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
# Bind the published reconstruction metadata when preparation has completed.
prepared = ROOT/'data/papers_g2_random_v2/prepared.json'
if prepared.exists():
 for name in ('prepared.json','sources.json','orders.json','hot_ready.json','traces_ready.json','admission.json',
              'valid_trace/manifest.json','test_trace/manifest.json','final/descriptor.json',
              'final/validation.json','final/bundle/manifest.json','final/payload_ready.json'):
  path = prepared.parent/name
  files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'ae/release_manifest.json').write_text(json.dumps({'schema':'digit-ae-release-v1','files':files},indent=2)+'\n')
print('Bound',len(files),'source/config files')
