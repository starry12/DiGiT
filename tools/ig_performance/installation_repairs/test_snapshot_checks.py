import hashlib,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from snapshot_checks import check_file
class Checks(unittest.TestCase):
    def check(self,source,uid=1000,mode=0o100664,same=True,ro=True,data=b'original'):
        with patch.object(Path,'stat',return_value=SimpleNamespace(st_uid=uid,st_mode=mode)),patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'read_bytes',return_value=data),patch('os.path.samefile',return_value=same),patch('os.statvfs',return_value=SimpleNamespace(f_flag=1 if ro else 0)):
            check_file('/release/config.json',hashlib.sha256(b'original').hexdigest(),source)
    def test_original_owner_readonly_input(self):self.check('/author/config.json')
    def test_root_code(self):self.check(None,uid=0,mode=0o100644)
    def test_code_owner_rejected(self):
        with self.assertRaises(AssertionError):self.check(None)
    def test_writable_code_rejected(self):
        with self.assertRaises(AssertionError):self.check(None,uid=0)
    def test_writable_mount_rejected(self):
        with self.assertRaises(AssertionError):self.check('/author/config.json',ro=False)
    def test_wrong_source_rejected(self):
        with self.assertRaises(AssertionError):self.check('/author/config.json',same=False)
    def test_changed_hash_rejected(self):
        with self.assertRaises(AssertionError):self.check('/author/config.json',data=b'changed')
if __name__=='__main__':unittest.main()
