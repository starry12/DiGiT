"""Exercise initial, partial, activated and concurrently changed deployment states."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import install
from install_support import record, atomic


class InstallStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(install, 'H', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.configs = [dict(key='ig_gcn', unit='model.service', selftest='check.service')]
        self.prepared = self.root / 'payload/ig_gcn/units'
        self.prepared.mkdir(parents=True)
        for name in ('model.service', 'check.service'):
            (self.prepared / name).write_text('[Service]\nType=oneshot\n')
            (self.prepared / name).chmod(0o664)
        self.live = self.root / 'live'
        self.live.mkdir()
        self.pointer = self.live / 'reviewer-current'
        self.pointer.symlink_to('/existing/reviewer')
        self.plan = dict(before={str(self.live / name): record(self.live / name)
                                for name in ('model.service', 'check.service', 'reviewer-current')})

    def copy_unit(self, name):
        atomic(self.live / name, (self.prepared / name).read_bytes())

    def test_first_install_then_activation(self):
        install.verify_server_state(self.plan, self.configs)
        for name in ('model.service', 'check.service'):
            self.copy_unit(name)
        install.verify_server_state(self.plan, self.configs, installed=True)

    def test_retry_partial_install_and_reject_missing_unit_at_activation(self):
        self.copy_unit('model.service')
        install.verify_server_state(self.plan, self.configs)
        with self.assertRaises(RuntimeError):
            install.verify_server_state(self.plan, self.configs, installed=True)

    def test_reject_concurrent_reviewer_change(self):
        self.pointer.unlink()
        self.pointer.symlink_to('/different/reviewer')
        with self.assertRaises(RuntimeError):
            install.verify_server_state(self.plan, self.configs)

    def test_reject_different_existing_unit(self):
        (self.live / 'model.service').write_text('unexpected unit')
        with self.assertRaises(RuntimeError):
            install.verify_server_state(self.plan, self.configs)

    def test_installed_mode_must_not_be_group_writable(self):
        for name in ('model.service', 'check.service'):
            self.copy_unit(name)
        (self.live / 'model.service').chmod(0o664)
        for installed in (False, True):
            with self.assertRaises(RuntimeError):
                install.verify_server_state(self.plan, self.configs, installed=installed)

    def test_staging_mode_does_not_change_destination_contract(self):
        for name in ('model.service', 'check.service'):
            self.copy_unit(name)
            (self.prepared / name).chmod(0o600)
        install.verify_server_state(self.plan, self.configs, installed=True)

    def test_reject_same_content_symlink(self):
        for name in ('model.service', 'check.service'):
            self.copy_unit(name)
        unit = self.live / 'model.service'
        unit.unlink()
        unit.symlink_to(self.prepared / 'model.service')
        with self.assertRaises(RuntimeError):
            install.verify_server_state(self.plan, self.configs, installed=True)


if __name__ == '__main__':
    unittest.main()
