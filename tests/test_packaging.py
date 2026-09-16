"""Check the Kaggle handoff without pretending to run GPU inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_kaggle import ROOT, run_pipeline


class PackagingTests(unittest.TestCase):
    def exercise(self, failed=False):
        calls=[]
        revision='a'*40
        def fake_run(command, **kwargs):
            self.assertEqual(kwargs['cwd'],ROOT)
            self.assertTrue(kwargs['check'])
            calls.append(command)
            if '--mode' in command and command[command.index('--mode')+1]=='smoke':
                out=Path(command[command.index('--out')+1])
                out.mkdir(parents=True,exist_ok=True)
                (out/'metrics.json').write_text(json.dumps({'counts':{'planned':4,'generation_successes':4,'prediction_failures':int(failed)}}))
                (out/'manifest.json').write_text(json.dumps({'resolved_revision':revision}))
        with tempfile.TemporaryDirectory() as tmp, patch('run_kaggle.subprocess.run',side_effect=fake_run):
            output=Path(tmp)/'new'/'runs'
            if failed:
                with self.assertRaisesRegex(RuntimeError,'pilot was not started'):
                    run_pipeline(output)
            else:
                self.assertEqual(run_pipeline(output),(output/'v1-pilot-qwen3-8b').resolve())
            self.assertTrue(output.is_dir())
        return calls,revision

    def test_smoke_then_pilot_same_model_revision_and_defaults(self):
        calls,revision=self.exercise()
        modes=[c[c.index('--mode')+1] for c in calls if '--mode' in c]
        self.assertEqual(modes,['smoke','pilot'])
        pilot=next(c for c in calls if 'pilot' in c)
        self.assertEqual(pilot[pilot.index('--revision')+1],revision)
        self.assertEqual(pilot[pilot.index('--model-size')+1],'8b')
        self.assertEqual(pilot[pilot.index('--cap')+1],'1536')

    def test_failed_smoke_blocks_pilot(self):
        calls,_=self.exercise(failed=True)
        self.assertFalse(any('pilot' in c for c in calls))


if __name__=='__main__':
    unittest.main()
