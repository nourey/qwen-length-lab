import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import (ROOT, count_output, derive_seed, run_phases, read_jsonl,
                        select_rows, validate_dataset, unique_index)
from metrics import (bucket, tails, parse_prediction, score, fit_baselines,
                     baseline_predictions, evaluate, bootstrap_delta, expected_tokens)


class MeasurementTests(unittest.TestCase):
    def test_strict_threshold_boundaries(self):
        self.assertEqual([bucket(n) for n in [1,128,129,256,257,512,513,1024,1025,1536]],
                         [0,0,1,1,2,2,3,3,4,4])

    def test_count_new_ids_eos_and_cap(self):
        self.assertEqual(count_output([7,8,99], [99], 10),
                         {"actual_tokens":3,"capped":False,"stop_reason":"eos"})
        self.assertFalse(count_output([7,99], [99], 2)["capped"])
        self.assertTrue(count_output([7,8], [99], 2)["capped"])
        with self.assertRaises(ValueError):
            count_output([], [99], 2)

    def test_probabilities_are_coherent(self):
        p, total = parse_prediction('{"bucket_probs":[0.1,0.2,0.3,0.15,0.25]}')
        for actual, expected in zip(tails(p), [.9,.7,.4,.25]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(expected_tokens([0,0,0,0,1],1536), 1280.5)
        p, total = parse_prediction('{"bucket_probs":[0.2,0.2,0.2,0.2,0.19]}')
        self.assertAlmostEqual(total,.99)
        self.assertAlmostEqual(sum(p),1)

    def test_bad_forecasts_remain_failures(self):
        for text in ['```json\n{"bucket_probs":[1,0,0,0,0]}\n```',
                     '{"bucket_probs":[0,0,0,0,0]}', '{"bucket_probs":[NaN,0,0,0,1]}',
                     '{"bucket_probs":[true,0,0,0,0]}', '{"bucket_probs":[1,0,0,0]}',
                     '{"bucket_probs":[1,0,0,0,0],"answer":0}',
                     '{"bucket_probs":[1,0,0,0,0],"bucket_probs":[0,1,0,0,0]}']:
            with self.subTest(text=text), self.assertRaises((ValueError,TypeError)):
                parse_prediction(text)

    def test_analytic_metrics(self):
        rows = [{"actual_tokens":128,"methods":{"m":[1,0,0,0,0]}},
                {"actual_tokens":129,"methods":{"m":[0,1,0,0,0]}}]
        result = score(rows,'m',1536)
        self.assertEqual(result['bucket_accuracy'],1)
        self.assertEqual(result['macro_threshold_brier'],0)
        self.assertEqual(result['mae_midpoint_tokens'],63.5)
        self.assertEqual(result['median_ae_midpoint_tokens'],63.5)
        rows[1]['methods']['m']=[1,0,0,0,0]
        result=score(rows,'m',1536)
        self.assertEqual(result['bucket_accuracy'],.5)
        self.assertEqual(result['thresholds'][0]['brier'],.5)
        self.assertEqual(result['macro_threshold_brier'],.125)
        self.assertEqual(result['confusion_matrix_actual_rows_predicted_columns'][1][0],1)

    def test_dev_only_baseline_and_degenerate_inputs(self):
        dev=[{'id':'d1','input_tokens':50,'actual_tokens':20},
             {'id':'d2','input_tokens':50,'actual_tokens':1500}]
        fitted=fit_baselines(dev,1536)
        test={'messages':[{'role':'user','content':'What is a queue?'}], 'input_tokens':1000, 'actual_tokens':10}
        p=baseline_predictions(test,fitted)
        test['actual_tokens']=1536
        self.assertEqual(p,baseline_predictions(test,fitted))
        self.assertTrue(all(math.isfinite(v) for v in p['prompt_length']))
        self.assertEqual(fitted['development_ids'],['d1','d2'])

    def test_dataset_and_seeds(self):
        rows=validate_dataset(read_jsonl(ROOT/'prompts.jsonl'))
        self.assertEqual(len(rows),48)
        self.assertEqual(sum(r['split']=='dev' for r in rows),16)
        self.assertEqual(sum(r['split']=='test' for r in rows),32)
        smoke=select_rows(validate_dataset(read_jsonl(ROOT/'smoke_prompts.jsonl')),'smoke')
        self.assertEqual(len(smoke),4)
        self.assertFalse({r['id'] for r in rows} & {r['id'] for r in smoke})
        self.assertFalse({json.dumps(r['messages'],sort_keys=True) for r in rows} &
                         {json.dumps(r['messages'],sort_keys=True) for r in smoke})
        self.assertEqual(len({json.dumps(r['messages'],sort_keys=True) for r in rows}),48)
        self.assertEqual(derive_seed(42,'a','generate'),derive_seed(42,'a','generate'))
        self.assertNotEqual(derive_seed(42,'a','generate'),derive_seed(42,'a','predict'))
        with self.assertRaises(ValueError):
            unique_index([{'id':'a'},{'id':'a'}],[{'id':'a'}])


class FakeRunner:
    eos_ids=[99]
    cap=1536
    def __init__(self,fail=False):
        self.calls=[]
        self.fail=fail
    def prediction_messages(self,row):
        return [{'role':'user','content':'STATE '+row['id']}],10
    def generate(self,messages,phase,seed):
        self.calls.append((phase,messages,seed))
        if phase=='predict':
            return {'text':'bad json' if self.fail else '{"bucket_probs":[0.2,0.2,0.2,0.2,0.2]}',
                    'output_ids':[1,2,99], 'input_tokens':15,'seconds':.01}
        return {'text':'OK','output_ids':[7,99],'input_tokens':10,'seconds':.02}


class PipelineTests(unittest.TestCase):
    def rows(self):
        return select_rows(validate_dataset(read_jsonl(ROOT/'smoke_prompts.jsonl')),'smoke')

    def test_freeze_before_generation_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            runner=FakeRunner()
            rows=run_phases(runner,self.rows(),path,42)
            self.assertEqual([c[0] for c in runner.calls],['predict']*4+['generate']*4)
            for call in runner.calls[4:]:
                self.assertNotIn('STATE ',json.dumps(call[1]))
            self.assertTrue((path/'predictions_frozen.json').exists())
            self.assertTrue(all(r['actual_tokens']==2 for r in rows))
            resumed=FakeRunner()
            self.assertEqual(rows,run_phases(resumed,self.rows(),path,42))
            self.assertEqual(resumed.calls,[])
            with (path/'predictions.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                run_phases(FakeRunner(),self.rows(),path,42)

    def test_invalid_predictions_do_not_drop_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            rows=run_phases(FakeRunner(fail=True),self.rows(),path,42)
            self.assertEqual(sum(r['prediction_status']=='error' for r in rows),4)
            self.assertEqual(sum(r['generation_status']=='ok' for r in rows),4)
            report=evaluate(rows,1536,path,'smoke')
            self.assertEqual(report['verdict'],'inconclusive')
            self.assertEqual(report['counts']['paired_test'],0)
            self.assertEqual(report['baseline_all_test']['dev_prior']['n'],2)

    def test_metrics_keep_failed_generation_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=run_phases(FakeRunner(),self.rows(),Path(tmp),42)
            rows[-1]['generation_status']='error'
            report=evaluate(rows,1536,tmp,'pilot')
            self.assertEqual(report['counts']['planned'],4)
            self.assertEqual(report['counts']['generation_failures'],1)
            self.assertEqual(report['verdict'],'inconclusive')

    def test_cluster_bootstrap_reproducibility(self):
        rows=[{'family':str(i),'actual_tokens':20,'methods':{'llm':[1,0,0,0,0],'baseline':[0,0,0,0,1]}} for i in range(3)]
        ci=bootstrap_delta(rows,'baseline',iterations=100)
        self.assertEqual(ci,bootstrap_delta(rows,'baseline',iterations=100))
        self.assertEqual(ci['percentile_95_interval'],[-1,-1])


if __name__=='__main__':
    unittest.main()
