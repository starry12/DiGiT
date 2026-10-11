"""Freeze once, before native GAT validation/full training."""
from candidates.pa_gat_native_v1.common import *

def main():
    require(not (HERE/'manifest.json').exists(),'Create a new version; do not re-sign')
    from submission.v3.common import verify as parent_verify
    parent=parent_verify()
    names=['atc26-paper1508.pdf','configs/papers/contract.json','ae/papers/runtime/models.py']
    # Transitive PA/SAGE + IO manifests preserve all reused binary/data helpers.
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    write(HERE/'manifest.json',dict(schema='digit-pa-gat-candidate-v1',parent_submission_sha256=parent,
        files=files,dependencies={n:sha(ROOT/n) for n in names},protocol_sha256=sha(P),
        classification='paper reconstruction; model-specific historical optimizer; not tuned on test',
        native_acceptance='requires separately verified preflight and fresh paired smoke'))
    print(json.dumps(dict(candidate_sha256=sha(HERE/'manifest.json'),files=len(files)),indent=2))
if __name__=='__main__':main()
