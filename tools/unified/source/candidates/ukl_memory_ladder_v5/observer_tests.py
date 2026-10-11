import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from . import observer as m
from . import start

class Tests(unittest.TestCase):
    def test_counters_errors(self):
        self.assertEqual(m.counters('write_bytes: 42\n'),{'write_bytes':42})
        self.assertEqual(m.counters({'error':'PermissionError'}),{'error':'PermissionError'})
    def test_attribution_pid_reuse_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proc=root/'proc';proc.mkdir();p=proc/'17';p.mkdir()
            def put(ticks,writes):
                (p/'stat').write_text('17 (worker odd) name) '+' '.join(['S']+['0']*18+[str(ticks)]+['0']*5))
                (p/'comm').write_text('worker\n');(p/'io').write_text('write_bytes: '+str(writes)+'\n')
                (p/'cgroup').write_text('0::/system.slice/test.service\n')
            cg=root/'cgroup';own=cg/'system.slice/test.service';own.mkdir(parents=True)
            (own/'memory.stat').write_text('file_dirty 4096\nfile_writeback 0\n')
            (own/'io.stat').write_text('259:0 wbytes=8192\n')
            stream=io.StringIO();o=m.Observer(stream,proc,cg);o.unit='test.service'
            put(10,100);self.assertIsNone(o.sample({})['processes'][0]['write_bytes_delta'])
            put(10,132);row=o.sample({});self.assertEqual(row['processes'][0]['write_bytes_delta'],32)
            self.assertEqual(row['cgroups'][-1]['memory']['file_dirty'],4096)
            put(11,20);self.assertIsNone(o.sample({})['processes'][0]['write_bytes_delta'])
            (p/'io').unlink();self.assertIn('error',o.sample({})['processes'][0]['io'])
            self.assertEqual(len(stream.getvalue().splitlines()),4)
            self.assertEqual(m.process(99,proc)['error'],'FileNotFoundError')
    def test_original_guards(self):
        before={'MemAvailable':100*1024**3,'Dirty':0,'Writeback':0}
        self.assertTrue(start.clean_memory(before,before))
        self.assertFalse(start.clean_memory(before,dict(before,Dirty=256*1024**2)))

if __name__=='__main__':unittest.main()
