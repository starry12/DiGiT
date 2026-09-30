import ast,json,tempfile,unittest
from dataclasses import fields
from pathlib import Path
from candidates.uks_native_v1.common import OUT as ORIGINAL,read
from candidates.uks_native_v1.storage import api
from .receipts import validate_saved_receipt
class ReceiptTests(unittest.TestCase):
    def test_both_real_receipts_accept_path(self):
        a=api()
        for arm in ('gids','digit'):
            folder=ORIGINAL/('storage_'+arm);saved=read(folder/'write_plan.json')
            plan=a.PayloadPlan(**{f.name:saved[f.name] for f in fields(a.PayloadPlan)})
            value=validate_saved_receipt(folder/a.VERIFY_RECEIPT,plan)
            self.assertEqual(value['status'],'verified')
            with self.assertRaises(TypeError):a.validate_verify_receipt(value,plan)
    def test_corrupt_receipt_is_rejected(self):
        a=api();folder=ORIGINAL/'storage_gids';saved=read(folder/'write_plan.json')
        plan=a.PayloadPlan(**{f.name:saved[f.name] for f in fields(a.PayloadPlan)})
        value=read(folder/a.VERIFY_RECEIPT)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'receipt.json'
            for key,bad in [('status','written_unverified'),('payload_bytes',0),('device_offset_bytes',0)]:
                path.write_text(json.dumps(dict(value,**{key:bad})))
                with self.assertRaises(RuntimeError):validate_saved_receipt(path,plan)
    def test_retry_exposes_only_smokes(self):
        from .controller import STAGES
        self.assertEqual(STAGES,('smoke_gids','smoke_digit'))
        from . import worker
        self.assertFalse(hasattr(worker,'profile'))
        tree=ast.parse(Path(worker.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and any(v.name in ('prepare','verify_payload') for v in n.names) for n in ast.walk(tree)))
    def test_teardown_error_not_accepted(self):
        from .controller import accept
        with self.assertRaises(ValueError):accept('smoke_gids',{'passed':True,'source_sha256':'x'},-6,'x')
if __name__=='__main__':unittest.main()
