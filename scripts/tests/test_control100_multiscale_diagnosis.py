from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'diagnose_b5_control100_multiscale.py'
spec = importlib.util.spec_from_file_location('multiscale_saved_diagnosis', SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def fixture(root):
    run, cache = root/'run', root/'cache'
    cache.mkdir(parents=True)
    (run/'shared').mkdir(parents=True)
    entries, cache_rows, arrays = [], {}, {}
    folders = {'initial':run/'adapt_control100/validation/epoch_0000',
               'B20':run/'adapt_control100/validation/epoch_0020',
               'C20':run/'adapt_multiscale/validation/epoch_0020'}
    for task, devices in (('task1',('watch_ecg',)), ('task2',('ecg_machine_d6','body_scale_d6'))):
        rows = []
        predictions = {m:[] for m in diag.METHODS}
        targets, masks = [], []
        for device in devices:
            cases = [('good0','one'), ('good1','one'), ('badI','one')]
            if task == 'task1':
                cases += [('good0','repeat')]
            for patient, repeat in cases:
                t = np.arange(60000)/500
                wave = (100*np.sin(t*9)+400*np.sin(t*.12)+50*np.sin(t*1.4)).astype(np.float32)
                y = np.tile(wave,(12,1))
                mask = np.ones(12,bool)
                mask[0] = patient != 'badI'
                mask[9] = patient != 'good1'  # good input, flagged V4 target.
                fm = (y - .7*diag.moving_average(y,501)).astype(np.float32)
                fm[0] = y[0]
                fixed_fast = fm.astype(float)-diag.moving_average(fm,501)
                true_slow = diag.moving_average(y,501)
                pair = device+'_'+patient+'_'+repeat
                key = task+':'+pair
                path = cache/(key+'.npz')
                # Colons are forbidden in Windows filenames; the logical key stays unchanged.
                path = cache/(key.replace(':','_')+'.npz')
                np.savez(path,fm_uV=fm,target_uV=y,mask=mask)
                entries.append({'cache_key':key,'target_record_id':patient,'subject_id':patient,
                                'file':path.name,'sha256':diag.sha(path)})
                p = {}
                for method, fraction in zip(diag.METHODS,(.40,.50,.65)):
                    p[method] = fixed_fast+.75*(fraction+(.01 if repeat=='repeat' else 0))*true_slow
                    p[method][0] = fm[0]
                for start in range(0,60000,5000):
                    rows.append({'pair_id':pair,'target_record_id':patient,'subject_id':patient,
                                 'input_type':device,'start_sample_500hz':str(start),'expected_window_count':'12'})
                    targets.append(y[:,start:start+5000])
                    masks.append(mask)
                    for method in diag.METHODS:
                        predictions[method].append(p[method][:,start:start+5000].astype(np.float32))
        cache_rows[task] = rows
        arrays[task] = {}
        for method in diag.METHODS:
            folder = folders[method]/task
            folder.mkdir(parents=True)
            p,y,q = np.stack(predictions[method]),np.stack(targets),np.stack(masks)
            for name,value in (('prediction_uV',p),('target_uV',y),('target_quality_mask',q)):
                np.save(folder/(name+'.npy'),value)
            diag.csv_save(folder/'window_metadata.csv',rows)
            overall,_ = diag.evaluate_record_predictions(p,y,task,rows,target_quality_mask=q,
                input_quality_mask=q[:,0],require_reliable_input=True,input_quality_source='same_record_target_I')
            arrays[task][method] = overall
    sampling = {'seed':42,'solver':'heun','steps':16,'samples':16}
    for method in diag.METHODS:
        score = diag.correlation_score_from_summaries({task:arrays[task][method] for task in arrays})
        diag.write(folders[method]/'evaluation.json',{'completed':True,
             'group':'adapt_multiscale' if method=='C20' else 'adapt_control100',
             'epoch':0 if method=='initial' else 20,'score':score,'sampling':sampling,'FM_frozen':True})
        diag.write(folders[method]/'experiment.json',{'beta':.75})
    diag.write(cache/'cache.json',{'completed':True,'sampling':sampling})
    study = {'protocol':'B5-control100-partner-multiscale-reliable-I-v2','score_protocol':diag.SCORE,
             'sampling':sampling,'beta':.75,'source_files':{str(cache/'cache.json'):diag.sha(cache/'cache.json')}}
    diag.write(run/'shared/study.json',study)
    diag.write(run/'shared/cache_view.json',{'completed':True,'study_sha256':diag.sha(run/'shared/study.json'),
        'validation_root':str(cache),'validation':entries,'validation_rows':cache_rows})
    for group in ('adapt_control100','adapt_multiscale'):
        diag.write(run/group/'initialization.json',{'study_sha256':diag.sha(run/'shared/study.json'),
            'cache_sha256':diag.sha(run/'shared/cache_view.json'),'score_protocol':diag.SCORE,
            'sampling':sampling,'beta':.75,'initial_slow_state_sha256':'same',
            'fixed_FM_state_sha256':'same','training':{'epochs':20}})
    return SimpleNamespace(run=str(run),output=str(root/'out'),initial_evaluation=None,b_evaluation=None,c_evaluation=None,
                           bootstrap=200,seed=42,plots=False,max_plots=6)


class DiagnosticTests(unittest.TestCase):
    def test_end_to_end_scores_qc_duplicate_weights_and_read_only_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            # B's windows are stored in a different row order; identity alignment must still work.
            folder = root/'run/adapt_control100/validation/epoch_0020/task1'
            metadata = diag.csv_read(folder/'window_metadata.csv')
            order = np.arange(len(metadata))[::-1]
            diag.csv_save(folder/'window_metadata.csv',[metadata[i] for i in order])
            for name in ('prediction_uV','target_uV','target_quality_mask'):
                path = folder/(name+'.npy')
                values = np.load(path,allow_pickle=False)[order].copy()
                np.save(path,values)
            files = [p for p in root.rglob('*') if p.is_file()]
            before = {str(p):diag.sha(p) for p in files}
            scores = diag.run(args)
            out = root/'out'
            report = diag.read(out/'diagnostic.json')
            self.assertTrue(report['completed'])
            self.assertFalse(report['model_loaded'])
            self.assertEqual(report['n_reliable_patients'],2)
            self.assertEqual(before,{str(p):diag.sha(p) for p in files})
            exclusions = diag.csv_read(out/'qc_exclusions.csv')
            self.assertEqual(sum(r['reason']=='actual_input_I_marked_unreliable' for r in exclusions),3)
            self.assertTrue(any(r['lead']=='V4' and r['reason']=='target_lead_marked_bad' for r in exclusions))
            target_rows = diag.csv_read(out/'target_lead_metrics.csv')
            self.assertTrue(all(r['target_record_id']!='badI' for r in target_rows))
            repeated = [r for r in target_rows if r['task']=='task1' and r['target_record_id']=='good0']
            self.assertTrue(all(int(r['n_pairs'])==2 for r in repeated))
            self.assertGreater(scores[-1]['main_correlation'],scores[1]['main_correlation'])
            self.assertEqual(len(diag.csv_read(out/'patient_diagnostics.csv')),2)
            with self.assertRaises(FileExistsError):
                diag.run(args)

    def test_actual_slow_recovery_and_approximate_band_identity(self):
        t = np.arange(60000)/500
        fm = np.sin(t*9)*100
        target = fm+300*np.sin(t*.2)
        desired = .75*150*np.sin(t*.2)
        prediction = fm-diag.moving_average(fm,501)+desired
        result = diag.metrics(prediction,target,fm,.75)
        self.assertAlmostEqual(result['slow_r'],diag.corr(desired,diag.moving_average(target,501)))
        np.testing.assert_allclose(sum(diag.bands(desired)),desired.reshape(6000,10).mean(-1),atol=1e-10)
        self.assertGreater(float(np.max(np.abs(desired-diag.moving_average(prediction,501)))),1)

    def test_subject_bootstrap_preserves_all_views_and_known_paired_delta(self):
        records = []
        for subject in range(4):
            for view in diag.VIEWS:
                for lead in diag.LEADS[1:]:
                    for method, gain in zip(diag.METHODS,(0,.01,.03)):
                        records.append({'subject_id':str(subject),'task':view[0],'input_type':view[1],
                            'target_record_id':str(subject),'lead':lead,'method':method,'raw_r':.3+.1*subject+gain})
        result = diag.bootstrap(records,200,43)
        self.assertEqual(result['n_patients'],4)
        self.assertAlmostEqual(result['C_minus_B']['point_delta_R'],.02)
        self.assertAlmostEqual(result['C_minus_B']['percentile_95_low'],.02)
        self.assertAlmostEqual(result['C_minus_B']['percentile_95_high'],.02)
        self.assertFalse(result['training_seed_replication'])

    def test_source_tampering_and_wrong_evaluation_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            view = diag.read(root/'run/shared/cache_view.json')
            path = root/'cache'/view['validation'][0]['file']
            path.write_bytes(b'changed archive')
            with self.assertRaisesRegex(ValueError,'cache entry changed'):
                diag.run(args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            path = root/'run/adapt_multiscale/validation/epoch_0020/evaluation.json'
            report = diag.read(path)
            report['score']['score_protocol'] = 'old-scope'
            diag.write(path,report)
            with self.assertRaisesRegex(ValueError,'Wrong epoch/group/score scope'):
                diag.run(args)


if __name__ == '__main__':
    unittest.main()
