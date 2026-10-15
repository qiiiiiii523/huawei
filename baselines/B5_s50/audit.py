from __future__ import annotations

import numpy as np
from .data import verify_inputs
from . import SCORE_PROTOCOL,FULL_SCORE_PROTOCOL


def verify_validation_inputs(datasets,metadata,arrays):
    pairs={}
    for pair in metadata['pairs']['task2']:
        if pair['target_record_id'] not in pairs or pair['input_type']=='ecg_machine_d6':
            pairs[pair['target_record_id']]=pair
    indices=[i for pair in pairs.values() for i in pair['indices']]
    base=datasets['task2']
    class View:
        rows=[base.rows[i] for i in indices]
        def __getitem__(self,index): return base[indices[index]]
    verify_inputs(View(),metadata['validation'],arrays)


def check_initial_score(scores,info,tolerance=2e-6,full=False):
    protocol=FULL_SCORE_PROTOCOL if full else SCORE_PROTOCOL
    if scores.get('competition_score_protocol')!=protocol: raise ValueError('Initializer score protocol differs')
    expected=info['source_full_summaries' if full else 'source_summaries']
    fields={'task1_r_missing11':float(expected['task1']['r_missing11']),
            'task2_r_missing11':float(expected['task2']['r_missing11']),
            'ecg_machine_d6_task2_missing_lead_mean_rmse_uV':float(expected['task2']['ecg_machine_d6_task2_missing_lead_mean_rmse_uV']),
            'body_scale_d6_task2_missing_lead_mean_rmse_uV':float(expected['task2']['body_scale_d6_task2_missing_lead_mean_rmse_uV'])}
    for key,value in fields.items():
        if not np.isclose(scores[key],value,atol=tolerance,rtol=tolerance):
            raise ValueError('Epoch-0 differs from specified best.pt reference: '+key)
