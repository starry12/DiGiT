"""Freeze code once, before preflight and native experiments."""
from candidates.ig_perf_v5.common import *
def main():
    require(not (HERE/'manifest.json').exists(),'Create a new version; never re-sign a frozen candidate')
    from submission.v4.common import verify as parent
    previous=parent();deps=[ROOT/'atc26-paper1508.pdf',ROOT/'digit_paths.py',ROOT/'ae/common.py',ROOT/'ae/release_manifest.json',ROOT/'configs/external_paths.json',ROOT/'configs/device.json',ROOT/'configs/ssd_payloads.json',ROOT/'ae/native/identity/identify-module',ROOT/'bam/build/lib/libnvm.so']
    # Freeze all imported IG Python/schema/native files. Parent v4 transitively
    # binds the useful-counter source headers and PA monitor/math/accounting helpers.
    for folder in ('ae/igb','configs/igb','candidates/ig_monitor_v1'):
        deps.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.so','.json'))
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(HERE/'manifest.json',dict(schema='digit-ig-perf-candidate-v1',parent_submission_sha256=previous,protocol_sha256=sha(P),files=files,dependencies={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(deps))},classification='paper reconstruction; IG 20 warmup plus 3x30 timed training batches only; no final accuracy/test claim',native_acceptance='requires preflight and fresh paired smoke before bounded benchmark'))
    print(json.dumps(dict(candidate_sha256=sha(HERE/'manifest.json'),files=len(files)),indent=2))
if __name__=='__main__':main()
