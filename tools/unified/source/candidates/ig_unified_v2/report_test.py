import copy,csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from candidates.unified_training_v2.config import load_config
from candidates.unified_training_v2.loop import run_arm
from candidates.unified_training_v2.tests import CPUAdapter,receipt
from .report import export

class Tests(unittest.TestCase):
    def test_ig_geometry_and_incomplete_rejection(self):
        config=load_config('IG','short1');ids=np.arange(320*1024+7,dtype=np.int64)
        rows=[receipt(run_arm(config,a,1,ids,CPUAdapter(ids))) for a in ('gids','digit')]
        series=dict(stage='complete',complete=True,passed=True,receipts=rows)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(ValueError):export(p,config,dict(series,complete=False))
            self.assertFalse((p/'summary.json').exists())
            result=export(p,config,series)
            self.assertEqual(result['row_bytes'],4096)
            self.assertEqual(result['evidence_scope'],'cpu_fixture')
            self.assertFalse(result['migration_acceptance_passed'])
            self.assertEqual(json.loads((p/'summary.json').read_text())['arms'],result['arms'])
            with (p/'summary.csv').open() as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),2)
            for row in table:
                self.assertEqual(int(row['primary_bytes']),4096*(int(row['primary_single_commands'])+2*int(row['primary_pair_commands'])))
            with self.assertRaises(ValueError):export(p,config,series)

if __name__=='__main__':unittest.main()
