"""Reuse an existing S50 original-loss control without changing its files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .context_audit import read, CANDIDATES
from .core import sha
from .losses import LOSS_PROTOCOL

PAIR_PROTOCOL = 'B5-S50-original-vs-F-v1'
REFERENCE = Path(__file__).resolve().parents[2] / 'configs/experiments/b5_exp_s50_f_reference.json'
FIXED_F_WEIGHT = 2.5890171538829843


def verify_unchanged_code(current):
    reference = read(REFERENCE)['implementation_sha256']
    for name, expected in reference.items():
        if name not in ('baselines/B5_exp/train_loss.py', 'baselines/B5_exp/check_loss.py') and current.get(name) != expected:
            raise ValueError('S50-F changed model, input, loss definition, scoring or runtime: ' + name)
    return reference


def canonical(config):
    if config.get('context_seconds') != 50 or config.get('mean_weight') != 0. or config.get('group') not in ('C', 'F'):
        raise ValueError('Expected S50 original loss or S50-F; mean loss must remain disabled')
    expected = 0. if config['group'] == 'C' else FIXED_F_WEIGHT
    if config.get('residual_weight') != expected:
        raise ValueError('F coefficient must be fixed before training')
    # Only loss activation and audited provenance differ; even inactive weight
    # metadata, seed, source checkpoint, prepared data and optimizer must agree.
    ignore = {'group', 'residual_weight', 'implementation_sha256', 'calibration_sha256',
              'context_extension_protocol', 'control_reference', 's50_f_protocol', 's50_f_control_reference'}
    return {k:v for k,v in config.items() if k not in ignore}


def audit_control(path, prepared_sha, init_sha, state_sha, b5_sha):
    path = Path(path).resolve()
    config = read(path / 'resolved_config.json')
    canonical(config)
    expected = {'protocol': 'B5-slow-probe-v1', 'experiment_protocol': LOSS_PROTOCOL, 'group':'C',
        'context_seconds':50, 'seed':42, 'epochs':20, 'batch_records':2, 'validate_every':5,
        'learning_rate':.0001, 'weight_decay':.01, 'huber_weight':1., 'pcc_weight':.1,
        'model_channels':48, 'prepared_sha256':prepared_sha, 'init_checkpoint_sha256':init_sha,
        'initial_weights_sha256':state_sha, 'source_epoch':50, 'mean_weight':0., 'residual_weight':0.,
        'residual_width_native':51, 'score_protocol':'correlation50_bonus10_target_record_v2',
        'epoch_zero_is_candidate':True, 'optimizer_initialization':'fresh_not_source_optimizer',
        'FM_training_started':False}
    for key,value in expected.items():
        if config.get(key) != value:
            raise ValueError('Existing S50 is not the matching control: ' + key)
    if config.get('implementation_sha256') != read(REFERENCE)['implementation_sha256']:
        raise ValueError('Unrecognized S50 control implementation; do not silently reuse it')
    if config['calibrated_weights']['residual'] != FIXED_F_WEIGHT:
        raise ValueError('Control has different fixed auxiliary-weight metadata')
    snapshot = path / 'context_preflight.json'
    if sha(snapshot) != config['calibration_sha256'] or not read(snapshot)['s50_compatibility'].get('ready'):
        raise ValueError('Invalid existing S50 preflight snapshot')
    completed = read(path / 'completed.json')
    if not completed.get('completed') or completed['additional_epochs'] != 20:
        raise ValueError('Existing S50 must have completed 20 additional epochs')
    import json
    history = [json.loads(line) for line in (path/'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [r['epoch'] for r in history] != list(range(21)):
        raise ValueError('Incomplete or duplicated S50 control history')
    initial = read(path/'initial_evaluation.json')
    assert_initial_equal(initial, history[0]['validation'])
    names = ['resolved_config.json','context_preflight.json','completed.json','initial_evaluation.json','history.jsonl']
    for candidate in CANDIDATES:
        evaluation = read(path / ('evaluation_'+candidate) / 'evaluation.json')
        if not evaluation.get('completed') or evaluation.get('group') != 'C' or evaluation.get('context_seconds') != 50:
            raise ValueError('Wrong S50 control evaluation: ' + candidate)
        if evaluation['probe_checkpoint_sha256'] != sha(path/(candidate+'.pt')) or evaluation['prepared_sha256'] != prepared_sha:
            raise ValueError('Stale S50 control evaluation: ' + candidate)
        if evaluation['B5_checkpoint_sha256'] != b5_sha:
            raise ValueError('Control used different fixed B5 predictions')
        if candidate == 'last' and evaluation['epoch'] != 20:
            raise ValueError('S50 control last is not epoch 20')
        names += [candidate+'.pt',f'evaluation_{candidate}/evaluation.json',f'evaluation_{candidate}/wave_summary.csv']
    return {'path':str(path),'files':{name:sha(path/name) for name in names},'B5_checkpoint_sha256':b5_sha}, config


def verify_control(proof):
    for name, expected in proof['files'].items():
        if sha(Path(proof['path'])/name) != expected:
            raise ValueError('S50 control changed after preflight: ' + name)


def assert_initial_equal(actual, reference):
    fields = ('task1_r_missing11','task2_r_missing11','ecg_machine_d6_task2_missing_lead_mean_rmse_uV',
              'body_scale_d6_task2_missing_lead_mean_rmse_uV','competition_total_score')
    for section in ('hybrid_score','baseline_score'):
        if not np.allclose([actual[section][k] for k in fields],[reference[section][k] for k in fields],rtol=0,atol=1e-6):
            raise ValueError('S50-F epoch-zero scores differ from matching S50 control: ' + section)
    key = 'slow_r_unique_quality_good_50Hz'
    if not np.isclose(actual[key],reference[key],rtol=0,atol=1e-6):
        raise ValueError('S50-F epoch-zero slow r differs from control')
