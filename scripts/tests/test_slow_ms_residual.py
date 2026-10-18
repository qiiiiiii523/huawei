"""CPU functional checks only: never calls optimizer.step or runs training."""
import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from baselines.B5 import ARCHITECTURE_ID, CONDITION_SCHEMA, PREPROCESSING_VERSION
from baselines.B5.config import ModelConfig
from baselines.B5.model import B5UNet
from baselines.B5.checkpoint import atomic_save, scales_digest
from baselines.B5.scoring import SCORE_PROTOCOL
from baselines.B5_best import SAMPLING, BUNDLE_PROTOCOL, SCORE_PROTOCOL as BUNDLE_SCORE_PROTOCOL
from baselines.B5_best.common import write, read, sha, state_sha
from baselines.B5_best.slow import SlowCNN, load_slow, predict_native
from baselines.B5_best.bundle import load_bundle
from baselines.B5_control_pcc import LEGACY_CACHE_PROTOCOL
from baselines.B5_control_pcc.core import generate, model_inputs
from baselines.B5_control_pcc.evaluate import evaluate as original_evaluate
from baselines.B5_control_pcc.losses import objective
from baselines.B5_slow_ms import PROTOCOL, CORRECTION, SLOW_PROTOCOL
from baselines.B5_slow_ms.model import SlowMSResidual
from baselines.B5_slow_ms.prepare import prepare
from baselines.B5_slow_ms.common import load_study, load_cache, make_model, load_checkpoint
from baselines.B5_slow_ms.evaluate import evaluate
from baselines.B5_slow_ms.export import export
from baselines.B5_slow_ms.summarize import summarize


def visible(length=2500):
    return dict(anchor=torch.sin(torch.arange(length).float()[None, None] / 80),
                numeric=torch.zeros(1, 3), sex=torch.zeros(1, dtype=torch.long),
                field_mask=torch.ones(1, 4), age_topcoded=torch.zeros(1, 1))


def bundle_fixture():
    cfg = ModelConfig(channels=(8, 8, 8), time_dim=8, metadata_dim=8, dilations=(1,))
    fm = B5UNet(cfg)
    scales = {'d12': [100.] * 12, 'ecg_machine_i': [100.]}
    state = fm.state_dict()
    slow = SlowCNN().state_dict()
    return {'bundle_protocol': BUNDLE_PROTOCOL, 'score_protocol': BUNDLE_SCORE_PROTOCOL,
            'sampling': dict(SAMPLING), 'slow_beta': .75,
            'fm': {'architecture_id': ARCHITECTURE_ID, 'architecture_hash': cfg.fingerprint,
                   'condition_schema': CONDITION_SCHEMA, 'preprocessing_version': PREPROCESSING_VERSION,
                   'model_config': asdict(cfg), 'scales': scales, 'scales_sha256': scales_digest(scales), 'ema': state},
            'slow': {'protocol': 'B5-slow-probe-v1', 'config': {'context_seconds': 50, 'group': 'F', 'model_channels': 48},
                     'model': slow}, 'fm_state_sha256': state_sha(state), 'slow_state_sha256': state_sha(slow),
            'provenance': {'slow_variant': 'slow_f', 'slow_sha256': 'fixture'}, 'reference_tolerance': 2e-5,
            'reference_qc_r': None, 'target_used_at_inference': False, 'prepared_data_required_at_inference': False}


def fixture(root):
    legacy, source = root / 'cache_run', root / 'source_run'
    cache = legacy / 'cache'
    cache.mkdir(parents=True)
    bundle = bundle_fixture()
    atomic_save(legacy / 'shared/components.pt', bundle)
    training = dict(epochs=20, batch_records=2, learning_rate=1e-4, weight_decay=.01,
                    seed=42, workers=0, gradient_clip=1., validate_every=5)
    source_base = {'bundle_sha256': sha(legacy / 'shared/components.pt'), 'training': training,
                   'train_records': 1, 'residual_weight': 2.5890171538829843,
                   'residual_width_native': 51, 'combo_weight': .1, 'mean_weight': 0., 'beta': .75}
    write(legacy / 'shared/study.json', source_base)
    entries = {}
    for key in ('train', 'good', 'good2', 'badI'):
        x = visible(6000)
        target = np.tile((100*np.sin(np.arange(60000)/65)+40*np.sin(np.arange(60000)/4000)).astype(np.float32), (12,1))
        mask = np.ones(12, bool)
        mask[0] = key != 'badI'
        mask[9] = key == 'good2'  # One flagged V4, another reliable target still scores V4.
        payload = {k: v[0].numpy() for k,v in x.items()}
        payload.update(target_uV=target, fm_uV=target*.8, mask=mask,
                       slow_target=target[1:].reshape(11,6000,10).mean(-1)/100)
        np.savez(cache / (key+'.npz'), **payload)
        entries[key] = dict(file=key+'.npz', sha256=sha(cache/(key+'.npz')), target_record_id=key,
                            subject_id=key, data_sha256=hashlib.sha256(key.encode()).hexdigest())
    validation, rows = [], {}
    for task, devices in (('task1', ['watch_ecg']), ('task2', ['ecg_machine_d6', 'body_scale_d6'])):
        task_rows = []
        for device in devices:
            for key in (['good','good','good2'] if task=='task1' else ['good','good2','badI']):
                pair = device+'_'+key+'_'+str(len(task_rows))
                item = dict(entries[key], cache_key=task+':'+pair)
                validation.append(item)
                for start in range(0,60000,5000):
                    task_rows.append(dict(pair_id=pair, target_record_id=key, subject_id=key,
                                          input_type=device, start_sample_500hz=str(start), expected_window_count='12'))
        rows[task] = task_rows
    manifest = {'protocol': LEGACY_CACHE_PROTOCOL, 'completed': True, 'FM_optimizer_updates': 0,
                'sampling': dict(SAMPLING), 'study_sha256': sha(legacy/'shared/study.json'),
                'train': [entries['train']], 'validation': validation, 'validation_rows': rows}
    write(cache / 'cache.json', manifest)
    atomic_save(source/'shared/components.pt', bundle)
    s = dict(source_base, protocol='B5-control100-partner-multiscale-reliable-I-v2',
             score_protocol=SCORE_PROTOCOL, sampling=dict(SAMPLING),
             source_files={str(cache/'cache.json'):sha(cache/'cache.json')})
    write(source/'shared/study.json', s)
    view = {'completed': True, 'study_sha256': sha(source/'shared/study.json'),
            'train_sources': {'adapt_multiscale': {'root': str(cache), 'entries': manifest['train']}},
            'validation_root': str(cache), 'validation': validation, 'validation_rows': rows}
    write(source/'shared/cache_view.json',view)
    # Score a synthetic full record with QC; this invokes no FM sampling or training.
    ev_study = dict(s, reference_score={'task1_r_missing11':0.,'task2_r_missing11':0.,'main_correlation':0.})
    base = load_slow(bundle['slow'],torch.device('cpu'),'slow_f')
    # Best weights deliberately differ from shared/components.pt's old initializer.
    with torch.no_grad():
        base.output.bias.add_(.2)
        base.output.weight.mul_(1.05)
    score = original_evaluate(base,bundle,ev_study,None,view,torch.device('cpu'),root/'reference','source',20)
    cp = dict(protocol=s['protocol'], group='adapt_multiscale', study_sha256=sha(source/'shared/study.json'),
              cache_sha256=sha(source/'shared/cache_view.json'), sampling_config=dict(SAMPLING),
              score_protocol=SCORE_PROTOCOL, epoch=20,best_epoch=20,best_r=score['main_correlation'],
              validation_score=score, fixed_FM_state_sha256=bundle['fm_state_sha256'],model=base.state_dict())
    atomic_save(source/'adapt_multiscale/best.pt',cp)
    return SimpleNamespace(source_run=str(source),source_checkpoint=None,control100_run=str(legacy),
                           training_seed=43,learning_rate=2e-5,epochs=20,output=str(root/'new/shared'))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_init_gradients_and_target_free_contract(self):
        torch.manual_seed(43)
        base = SlowCNN().eval()
        initial = state_sha(base.state_dict())
        model = SlowMSResidual(copy.deepcopy(base)).train()
        self.assertFalse(model.base.training)
        x = visible()
        with torch.no_grad():
            expected = base(**x)
        output = model(**x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        (output-torch.randn_like(output)).square().mean().backward()
        self.assertTrue(model.correction.output.weight.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.base.parameters()))
        self.assertEqual(initial,state_sha(model.base.state_dict()))
        model.zero_grad(set_to_none=True)
        # Synthetic weight assignment verifies interior gradients, without optimizer updates.
        with torch.no_grad():
            model.correction.output.weight.fill_(.001)
        model(**x).square().mean().backward()
        self.assertTrue(model.correction.input.conv.weight.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        full = visible(6000)
        full['target_uV'] = torch.full((1,12,60000), float('nan'))
        full['slow_target'] = torch.full((1,11,6000),float('nan'))
        self.assertEqual(set(model_inputs(full)), {'anchor','numeric','sex','field_mask','age_topcoded'})
        model.eval()
        with torch.no_grad():
            a = generate(model,model_inputs(full))
            b = predict_native(model,model_inputs(full))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_prepare_qc_checkpoint_export_and_immutability(self):
        with tempfile.TemporaryDirectory() as temp, patch('torch.optim.AdamW.step',side_effect=AssertionError('No local training')):
            root=Path(temp)
            args=fixture(root)
            tracked=list((root/'source_run').rglob('*'))+list((root/'cache_run').rglob('*'))
            before={str(p):sha(p) for p in tracked if p.is_file()}
            report=prepare(args)
            self.assertTrue(report['ready'])
            self.assertEqual(report['zero_init_max_abs_difference'],0.)
            self.assertEqual(report['source_best_epoch'],20)
            self.assertEqual(report['validation_audit']['task2']['ecg_machine_d6']['eligible_pairs'],2)
            run=root/'new'; study_path=run/'shared/study.json'; cache_path=run/'shared/cache_view.json'
            study,bundle=load_study(study_path)
            self.assertNotEqual(bundle['slow_state_sha256'],load_bundle(root/'source_run/shared/components.pt')['slow_state_sha256'])
            view=load_cache(cache_path,study_path)
            self.assertEqual(study['training']['learning_rate'],2e-5)
            self.assertEqual(study['extra_scale_loss_weight'],0.)
            for group in ('control','residual'):
                model=make_model(bundle,group,torch.device('cpu')).eval()
                folder=run/group/'evaluation_best_test'
                score=evaluate(model,bundle,study,cache_path,view,torch.device('cpu'),folder,group,0,initial=True)
                self.assertTrue(read(folder/'evaluation.json')['reference_matched'])
                cp=dict(protocol=PROTOCOL,group=group,study_sha256=sha(study_path),cache_sha256=sha(cache_path),
                        sampling_config=dict(SAMPLING),score_protocol=SCORE_PROTOCOL,extra_scale_loss_weight=0.,
                        fixed_FM_state_sha256=bundle['fm_state_sha256'],epoch=0,model=model.state_dict(),validation_score=score)
                atomic_save(run/group/'best.pt',cp)
                load_checkpoint(run/group/'best.pt',study_path,torch.device('cpu'))
                portable=export(study_path,run/group/'best.pt',run/group/'portable.pt')
                loaded=load_bundle(run/group/'portable.pt')
                deployed=load_slow(loaded['slow'],torch.device('cpu'),'slow_f')
                with torch.no_grad():
                    torch.testing.assert_close(generate(model,visible(6000)),predict_native(deployed,visible(6000)),rtol=0,atol=0)
                self.assertEqual(portable['fm_state_sha256'],bundle['fm_state_sha256'])
                write(run/group/'completed.json',dict(completed=True))
                for name in ('best','last'):
                    (run/group/f'latest_{name}_evaluation.txt').write_text(str(folder))
            table=summarize(run)
            self.assertEqual(len(table),5)
            self.assertTrue(all(abs(r['delta_vs_initializer'])<2e-5 for r in table))
            self.assertEqual(before,{str(p):sha(p) for p in tracked if p.is_file()})
            # A modified cache is rejected before any training/evaluation.
            p=root/'cache_run/cache/train.npz'
            p.write_bytes(p.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):
                load_cache(cache_path,study_path)

    def test_portable_target_free_predictor_with_nonzero_correction(self):
        from baselines.B5_best.predict import Predictor
        from baselines.B5_best.combine import combine, upsample_native
        model=SlowMSResidual().eval()
        with torch.no_grad():
            model.correction.output.weight.fill_(.0001)
            model.correction.output.bias.fill_(.05)
        bundle=bundle_fixture()
        bundle['slow']={'protocol':SLOW_PROTOCOL,'config':dict(context_seconds=50,group='F',model_channels=48,
                        correction=copy.deepcopy(CORRECTION)), 'model':model.state_dict()}
        bundle['slow_state_sha256']=state_sha(bundle['slow']['model'])
        raw=(100*np.sin(np.arange(60000)/700)).astype(np.float32)[None]
        metadata={k:v[0].numpy() for k,v in visible().items() if k!='anchor'}
        def fake_sample(fm, inputs, keys, **sampling):
            self.assertEqual(sampling,SAMPLING)
            self.assertTrue(all(k.startswith('new_patient:') for k in keys))
            return inputs['anchor'].repeat(1,12,1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'portable.pt'
            atomic_save(path,bundle)
            loaded=load_bundle(path)
            with patch('baselines.B5_best.predict.sample',side_effect=fake_sample), \
                 patch('baselines.B5_slow_ms.common.load_study',side_effect=AssertionError('Inference must not load study/cache')):
                predictor=Predictor(loaded,torch.device('cpu'))
                output=predictor.predict_encoded(raw,metadata,'new_patient',batch_size=12)
            inputs={k:torch.from_numpy(v[None].copy()) for k,v in metadata.items()}
            inputs['anchor']=torch.from_numpy((raw.reshape(1,6000,10).mean(-1)/100)[None])
            with torch.no_grad():
                native=generate(model,inputs)[0].numpy()
            learned=upsample_native(native.astype(np.float64)*100.)
            expected=combine(np.tile(raw,(12,1)),learned,.75)
            np.testing.assert_allclose(output,expected,rtol=2e-6,atol=1e-4)
            # Existing FM normalization/restore is float32; preserve I to rounding precision.
            np.testing.assert_allclose(output[0],raw[0],rtol=1e-6,atol=1e-5)

    def test_training_requires_explicit_execute_flag(self):
        from baselines.B5_slow_ms.train import main
        with patch('sys.argv',['train','--study','missing','--cache','missing','--output','missing','--group','residual']), patch('baselines.B5_slow_ms.train.run') as run:
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code,2)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
