import unittest
from .preparation_compatibility import compatible_files, PREFIX


class PreparationCompatibility(unittest.TestCase):
    def test_only_explicit_monitor_revision_changes(self):
        old = {PREFIX+'ig/controller.py': 'a', PREFIX+'native/reuse.cpp': 'b',
               'results/group_cache_migration_v1/build/reuse': 'c'}
        new = dict(old)
        new[PREFIX+'ig/controller.py'] = 'changed'
        self.assertTrue(compatible_files(old, new))
        new[PREFIX+'native/reuse.cpp'] = 'changed'
        self.assertFalse(compatible_files(old, new))
        new = dict(old)
        del new['results/group_cache_migration_v1/build/reuse']
        self.assertFalse(compatible_files(old, new))
