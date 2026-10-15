"""Audit the committed C reference and unchanged S30 model/scoring contracts."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from .core import sha
from .losses import LOSS_PROTOCOL

CONTEXT_PROTOCOL = 'B5-S30-S50-original-loss-v1'
REFERENCE_FILE = Path(__file__).resolve().parents[2] / 'configs/experiments/b5_exp_s50_reference.json'
CANDIDATES = ('best_slow_r', 'best_score', 'last')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_code_hash(source):
    tree = ast.parse(source)
    nodes = []
    for node in tree.body:
        wanted = isinstance(node, ast.ClassDef) and node.name in ('SameConv', 'Block', 'SlowCNN')
        wanted |= isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DILATIONS' for t in node.targets)
        if wanted:
            if isinstance(node, ast.ClassDef) and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]  # Docstrings do not change the model computation.
            nodes.append(node)
    if len(nodes) != 4:
        raise ValueError('Incomplete CNN definition')
    module = ast.Module(body=nodes, type_ignores=[])
    for node in ast.walk(module):
        # Python 3.12 added type_params even for non-generic classes/functions.
        # Ignore that empty syntax field so Windows and Python 3.10 server agree.
        node._fields = tuple(field for field in node._fields if field != 'type_params')
    return hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()


def verify_model_and_fixed_code(current):
    reference = read(REFERENCE_FILE)
    package = Path(__file__).resolve().parent
    if model_code_hash((package / 'core.py').read_text(encoding='utf-8')) != reference['model_code_sha256']:
        raise ValueError('CNN definition changed; not an input-context-only experiment')
    mutable = {'baselines/B5_exp/core.py', 'baselines/B5_exp/train_loss.py', 'baselines/B5_exp/check_loss.py'}
    for name, expected in reference['implementation_sha256'].items():
        if name not in mutable and current.get(name) != expected:
            raise ValueError('Loss, evaluation or runtime changed from C reference: ' + name)
    return reference


def canonical(config):
    # Both active auxiliary coefficients must be zero before ignoring inactive weights.
    if config.get('group') != 'C' or config.get('mean_weight') != 0 or config.get('residual_weight') != 0:
        raise ValueError('Context ablation requires the original loss in both runs')
    ignore = {'context_seconds', 'implementation_sha256', 'calibration_sha256', 'calibrated_weights',
              'context_extension_protocol', 'control_reference'}
    return {k: v for k, v in config.items() if k not in ignore}


def audit_control(path, prepared_sha, init_sha, initial_state_sha):
    path = Path(path).resolve()
    config = read(path / 'resolved_config.json')
    reference = read(REFERENCE_FILE)
    expected = {'experiment_protocol': LOSS_PROTOCOL, 'context_seconds': 30, 'group': 'C',
                'mean_weight': 0., 'residual_weight': 0., 'seed': 42, 'epochs': 20, 'batch_records': 2,
                'validate_every': 5, 'learning_rate': .0001, 'weight_decay': .01, 'huber_weight': 1.,
                'pcc_weight': .1, 'model_channels': 48, 'prepared_sha256': prepared_sha,
                'init_checkpoint_sha256': init_sha, 'initial_weights_sha256': initial_state_sha,
                'epoch_zero_is_candidate': True, 'optimizer_initialization': 'fresh_not_source_optimizer',
                'score_protocol': 'correlation50_bonus10_target_record_v2', 'FM_training_started': False}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Existing C run is not the matching S30 control: ' + key)
    if config.get('implementation_sha256') != reference['implementation_sha256']:
        raise ValueError('Existing C run used an unrecognized implementation; do not silently reuse it')
    completed = read(path / 'completed.json')
    if not completed.get('completed') or completed['additional_epochs'] != 20:
        raise ValueError('C control did not finish 20 additional epochs')
    history = [json.loads(line) for line in (path / 'history.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if [row['epoch'] for row in history] != list(range(21)):
        raise ValueError('Incomplete C control history')
    names = ['resolved_config.json', 'completed.json', 'initial_evaluation.json', 'history.jsonl']
    b5_sha = None
    for candidate in CANDIDATES:
        folder = path / ('evaluation_' + candidate)
        evaluation = read(folder / 'evaluation.json')
        if not evaluation.get('completed') or evaluation.get('group') != 'C' or evaluation.get('context_seconds') != 30:
            raise ValueError('Wrong C evaluation: ' + candidate)
        if evaluation['probe_checkpoint_sha256'] != sha(path / (candidate + '.pt')) or evaluation['prepared_sha256'] != prepared_sha:
            raise ValueError('Stale C evaluation: ' + candidate)
        if candidate == 'last' and evaluation['epoch'] != 20:
            raise ValueError('C last.pt is not epoch 20')
        if b5_sha is not None and evaluation['B5_checkpoint_sha256'] != b5_sha:
            raise ValueError('C candidates used different B5 predictions')
        b5_sha = evaluation['B5_checkpoint_sha256']
        names += [candidate + '.pt', f'evaluation_{candidate}/evaluation.json', f'evaluation_{candidate}/wave_summary.csv']
    return {'path': str(path), 'files': {name: sha(path / name) for name in names}, 'B5_checkpoint_sha256': b5_sha}, config


def verify_control_files(proof):
    for name, expected in proof['files'].items():
        if sha(Path(proof['path']) / name) != expected:
            raise ValueError('C control changed after S50 preflight: ' + name)
