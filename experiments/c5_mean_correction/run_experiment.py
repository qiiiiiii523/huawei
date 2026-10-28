"""Isolated C5 EMA1s mean-correction experiment; immutable source model/data."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
import torch

from mean_model import (MeanCorrector, GENERATED, GROUP_WEIGHTS, LEADS,
                        make_example, offsets11, statistics_metrics, learning_loss)

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('C5_PROJECT',str(HERE.parents[1])))
QUEUE = Path(os.environ.get('C5_QUEUE','/home/qht/train_scheduler'))
CHECKPOINT = Path(os.environ.get('C5_BASE_CHECKPOINT',str(QUEUE/'runs/20261027-093153-e745e320/experiment/A/best_r.pt')))
VIEWS = Path(os.environ.get('C5_VIEWS','/home/qht/huawei/data_cache/B7_C_round1/huawei10/manifest.json'))
VALIDATION_CACHE = Path(os.environ.get('C5_VALIDATION_CACHE',str(QUEUE/'runs/20261027-142425-7dc8cea6/cache')))
FULL_VALIDATION = Path(os.environ.get('C5_FULL_VALIDATION',str(QUEUE/'C_full120_20261027/full120_manifest.json')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def project_imports():
    sys.path.insert(0, str(PROJECT))
    from baselines.B7.data import read_manifest, load_record
    from baselines.B7.checkpoint import load_checkpoint, model_from_checkpoint
    from baselines.B7.config import SamplingConfig
    from baselines.B7.inference import predict_records
    return read_manifest, load_record, load_checkpoint, model_from_checkpoint, SamplingConfig, predict_records


def sources():
    read_manifest, _, load_checkpoint, _, SamplingConfig, _ = project_imports()
    views = read_manifest(VIEWS)
    source = Path(views['view_source_manifest'])
    if sha(source) != views['view_source_sha256']:
        raise ValueError('Original full120 source manifest changed')
    manifest = read_manifest(source)
    rows = [r for r in manifest['records'] if r['split'] in ('train','validation') and r['input_reliable']]
    if any('window_start' in r for r in rows):
        raise ValueError('Expected original full records, not10s view records')
    train = [r for r in rows if r['split']=='train']
    val = [r for r in rows if r['split']=='validation']
    if len(train)!=359 or len(val)!=90:
        raise ValueError(('Source membership changed', len(train), len(val)))
    if set(r['subject_id'] for r in train) & set(r['subject_id'] for r in val):
        raise ValueError('Training/validation subject leakage')
    ck = load_checkpoint(CHECKPOINT)
    sampling = SamplingConfig(**ck['sampling'])
    if (sampling.window_seconds, sampling.core_seconds, sampling.steps, sampling.samples,
            sampling.validation_batch_size, sampling.sample_batch_size)!=(10,10,16,16,4,16):
        raise ValueError('Frozen C5 sampling changed')
    if list(ck['scales_uV']) != views['scales_uV']:
        raise ValueError('Frozen train-only scales changed')
    old = json.loads((VALIDATION_CACHE/'completed.json').read_text())
    identity = old['identity']
    if (not old['completed'] or identity['mode']!='correct'
            or identity['checkpoint_sha256']!=sha(CHECKPOINT)
            or identity['manifest_sha256']!=sha(FULL_VALIDATION)
            or identity['sampling']!=ck['sampling']
            or [r['record_id'] for r in val]!=identity['records']):
        raise ValueError('Existing validation cache does not match A/B')
    spec = dict(protocol='C5-frozen-EMA-mean-correction-1s-v1', checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha(CHECKPOINT), views_sha256=sha(VIEWS), source=str(source), source_sha256=sha(source),
        full_validation_sha256=sha(FULL_VALIDATION), validation_cache=str(VALIDATION_CACHE),
        scales_uV=ck['scales_uV'], sampling=ck['sampling'],
        train_ids=[r['record_id'] for r in train], validation_ids=[r['record_id'] for r in val],
        code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},
        network='9->32 Conv5/GELU->32 Conv5/GELU->AvgPool50->4x(2Conv3 residual,dilation1,2,4,8)->7 Conv1 zero init',
        input='50Hz block averages of measured I + seven frozen predictions, fixed train-only scales, absolute record position; no target inputs',
        output='7 offsets per1s, repeat500samples, derive other limb leads from fixed I and corrected II',
        objective='QC/physical-target/group/lead weighted smoothL1 normalized mean error +0.1 full120 PCC loss',
        training=dict(seed=42, epochs=100, batch_size=16, lr=.001, weight_decay=.0001,
                      validation_every=5, best_selection='full120_original_missing11_r_including_zero_correction',
                      precision='float32 model, float64 moments', base_updates=0),
        train_subjects=len({r['subject_id'] for r in train}), validation_subjects=len({r['subject_id'] for r in val}),
        baseline_r=old['metrics']['r'])
    return spec, train, val, ck, old


def checked_plan():
    spec, train, val, ck, old = sources()
    plan = json.loads((HERE/'plan.json').read_text())
    if plan != spec:
        raise ValueError('Frozen experimental code/config/data changed; create a new run')
    return spec, train, val, ck, old


def trace_model(model):
    for name in ('_jit_override_can_fuse_on_gpu','_jit_override_can_fuse_on_cpu','_jit_set_texpr_fuser_enabled'):
        if hasattr(torch._C, name):
            getattr(torch._C, name)(False)
    device = next(model.parameters()).device
    state = torch.zeros(16,7,5000,device=device)
    anchor = torch.zeros(16,1,5000,device=device)
    md = dict(numeric=torch.zeros(16,3,device=device), sex=torch.full((16,),2,device=device,dtype=torch.long),
        field_mask=torch.zeros(16,4,device=device,dtype=torch.bool), age_topcoded=torch.zeros(16,1,device=device),
        record_position=(torch.arange(5000,device=device)/30000-1)[None,None,:].expand(16,1,5000))
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.jit.TracerWarning)
        traced = torch.jit.trace(model, (state,torch.linspace(0,1,16,device=device),anchor,md), check_trace=False, strict=True)
    class Runner(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.traced=traced; self.config=model.config
        def forward(self,state,t,anchor,metadata):
            return self.traced(state,t,anchor,metadata)
    return Runner().eval()


def configure():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def cache_records(args):
    spec, train, val, ck, old = checked_plan()
    read_manifest, load_record, _, model_from_checkpoint, SamplingConfig, predict_records = project_imports()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write(out/'identity.json', dict(plan=spec, shard=args.shard, shards=args.shards))
    configure()
    selected = train[args.shard::args.shards]
    if not selected:
        raise ValueError('Empty shard')
    model = model_from_checkpoint(ck, 'cuda')
    model.requires_grad_(False)
    sampling = SamplingConfig(**ck['sampling'])
    def infer(m, rows):
        records=[load_record(r) for r in rows]
        with torch.inference_mode():
            return predict_records(m,[v[0] for v in records],ck['scales_uV'],
                [r['record_id'] for r in rows],[v[3] for v in records],sampling).cpu().numpy()
    ref = infer(model, selected[:1])
    model = trace_model(model)
    candidate = infer(model, selected[:1])
    difference=float(np.max(abs(ref-candidate)))
    if difference>.1:
        raise ValueError(('Trace/eager disagreement',difference))
    write(out/'trace_check.json', dict(max_abs_uV=difference, passed=True))
    entries=[]
    begin=time.monotonic()
    for st in range(0,len(selected),4):
        chunk=selected[st:st+4]
        before=[sha(r['resolved_path']) for r in chunk]
        predictions=infer(model,chunk)
        for j,row in enumerate(chunk):
            anchor,target,mask,_=load_record(row)
            if sha(row['resolved_path'])!=before[j]:
                raise ValueError('Source waveform changed during inference')
            name=hashlib.sha256(row['record_id'].encode()).hexdigest()[:24]+'.npz'
            example=make_example(anchor,predictions[j,list(GENERATED)],target,mask,ck['scales_uV'])
            np.savez_compressed(out/name,**example)
            entries.append(dict(row={k:row[k] for k in ('record_id','target_record_id','subject_id','split','group')},
                                file=name, source_sha256=before[j], example_sha256=sha(out/name)))
        write(out/'progress.json',dict(records=len(entries),total=len(selected),seconds=time.monotonic()-begin))
        print('CACHE',args.shard,len(entries),'/',len(selected),'seconds',round(time.monotonic()-begin),flush=True)
    write(out/'completed.json',dict(completed=True,plan=spec,shard=args.shard,shards=args.shards,entries=entries,
        seconds=time.monotonic()-begin,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))


def load_examples(shards, ck, val, old):
    _, load_record, *_=project_imports()
    examples=[]; rows=[]; seen=set()
    plan=json.loads((HERE/'plan.json').read_text())
    for directory in shards:
        done=json.loads((directory/'completed.json').read_text())
        if not done['completed'] or done['plan']!=plan:
            raise ValueError('Incomplete or incompatible training cache')
        for entry in done['entries']:
            row=entry['row']
            if row['record_id'] in seen or row['split']!='train':
                raise ValueError('Duplicate/nontrain training example')
            seen.add(row['record_id'])
            f=directory/entry['file']
            if sha(f)!=entry['example_sha256']:
                raise ValueError('Cached example changed')
            with np.load(f, allow_pickle=False) as d:
                examples.append({k:d[k].copy() for k in d.files})
            rows.append(row)
    if seen!=set(plan['train_ids']):
        raise ValueError('Training cache membership mismatch')
    validation=[]
    oldp=np.load(VALIDATION_CACHE/'predictions.npy',mmap_mode='r')
    expected={r['record_id']:r for r in old['records']}
    maxdiff=0.
    for idx,row in enumerate(val):
        if sha(row['resolved_path'])!=old['identity']['raw_sha'][row['record_id']]:
            raise ValueError('Validation target changed since frozen A/B cache')
        anchor,y,mask,_=load_record(row)
        item=make_example(anchor,np.asarray(oldp[idx]),y,mask,ck['scales_uV'])
        tensor={k:torch.from_numpy(v)[None] for k,v in item.items()}
        r,_=statistics_metrics(*(tensor[k] for k in ('pm','ym','pss','yss','cross')),torch.zeros_like(tensor['pm']))
        for j in np.flatnonzero(item['mask']):
            original=expected[row['record_id']]['leads'][str(j+1)]['r']
            if original is not None:
                maxdiff=max(maxdiff,abs(float(r[0,j])-original))
        validation.append(item)
    if maxdiff>2e-5:
        raise ValueError(('Exact full120 sufficient-statistics mismatch',maxdiff))
    return examples,rows,validation,maxdiff


def pack(examples, rows):
    tensors={k:torch.from_numpy(np.stack([x[k] for x in examples])) for k in examples[0]}
    # Equal physical targets within group/lead; average repeated pairs first.
    cells=Counter((r['group'],j,r['target_record_id']) for r,e in zip(rows,examples)
                  for j in np.flatnonzero(e['mask']))
    targets=Counter((g,j) for g,j,_ in cells)
    weights=np.zeros((len(rows),11),dtype=np.float64)
    for i,(row,ex) in enumerate(zip(rows,examples)):
        for j in np.flatnonzero(ex['mask']):
            key=(row['group'],j,row['target_record_id'])
            weights[i,j]=GROUP_WEIGHTS[row['group']]/11/targets[key[:2]]/cells[key]
    tensors['weights']=torch.from_numpy(weights)
    return tensors


def aggregate(values, rows):
    cells=defaultdict(list)
    for row,v in zip(rows,values):
        for j,x in enumerate(v):
            if np.isfinite(x):
                cells[(row['group'],j,row['target_record_id'])].append(float(x))
    groups={}; leads={}
    for g in GROUP_WEIGHTS:
        leads[g]={}
        for j,name in enumerate(LEADS):
            items=[np.mean(v) for (gg,jj,_),v in cells.items() if gg==g and jj==j]
            if not items:
                raise ValueError(('Empty scoring cell',g,name))
            leads[g][name]=float(np.mean(items))
        groups[g]=float(np.mean(list(leads[g].values())))
    perlead={name:sum(GROUP_WEIGHTS[g]*leads[g][name] for g in GROUP_WEIGHTS) for name in LEADS}
    return dict(all11=sum(GROUP_WEIGHTS[g]*groups[g] for g in GROUP_WEIGHTS),groups=groups,
        lead_groups=leads,leads=perlead,chest6=float(np.mean([perlead[n] for n in LEADS[5:]])),
        limb5=float(np.mean([perlead[n] for n in LEADS[:5]])))


def score(delta7, data, rows, scales):
    with torch.no_grad():
        delta11=offsets11(delta7.double()*scales[list(GENERATED)][None,:,None])
        r,rmse=statistics_metrics(*(data[k] for k in ('pm','ym','pss','yss','cross')),delta11)
        before=data['pm']-data['ym']
        after=before+delta11
        mrms=after.square().mean(-1).sqrt()
        trend=(after-after.mean(-1,keepdim=True)).square().mean(-1).sqrt()
        arrays=[a.numpy().copy() for a in (r,rmse,mrms,trend)]
        for a in arrays:
            a[~data['mask'].numpy()]=np.nan
        result={k:aggregate(a,rows) for k,a in zip(('r','rmse_uV','mean_error_rms_uV','nonDC_mean_error_rms_uV'),arrays)}
        return result,arrays


def forward_all(model, data):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(data['features'][st:st+16]) for st in range(0,len(data['features']),16)])


def train_corrector(args):
    spec, train, val, ck, old=checked_plan()
    configure()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    write(out/'specification.json',spec)
    examples,rows,validation,maxdiff=load_examples(args.shard_dirs,ck,val,old)
    data=pack(examples,rows);vd=pack(validation,val)
    scales=torch.tensor(ck['scales_uV'],dtype=torch.float64)
    model=MeanCorrector().cpu()
    count=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    zeros=torch.zeros(len(val),7,120)
    baseline,baseline_arrays=score(zeros,vd,val,scales)
    if abs(baseline['r']['all11']-spec['baseline_r'])>2e-5:
        raise ValueError('A baseline mismatch')
    # Matched structural oracle preserves fixed I and the original limb algebra.
    oracle7=(vd['ym'][:,[0,5,6,7,8,9,10]]-vd['pm'][:,[0,5,6,7,8,9,10]])/scales[list(GENERATED)][None,:,None]
    oracle,_=score(oracle7,vd,val,scales)
    all11_r,all11_rmse=statistics_metrics(*(vd[k] for k in ('pm','ym','pss','yss','cross')),vd['ym']-vd['pm'])
    all11_r=all11_r.numpy();all11_r[~vd['mask'].numpy()]=np.nan
    b_unconstrained=aggregate(all11_r,val)
    best=baseline['r']['all11'];best_epoch=0;history=[]
    best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    write(out/'baseline_and_oracles.json',dict(A=baseline,B_structural_1s=oracle,
        B_independent11_1s_r=b_unconstrained,exact_statistics_max_reference_r_difference=maxdiff,
        parameters=count,training_pairs=len(rows),validation_pairs=len(val)))
    begin=time.monotonic()
    for epoch in range(1,101):
        model.train();permutation=torch.randperm(len(rows));losses=[]
        for start in range(0,len(rows),16):
            indices=permutation[start:start+16]
            batch={k:v[indices] for k,v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch['features'])
            loss=learning_loss(batch,prediction,scales)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        if epoch%5==0:
            valid,_=score(forward_all(model,vd),vd,val,scales)
            train_score,_=score(forward_all(model,data),data,rows,scales)
            rec=dict(epoch=epoch,loss=float(np.mean(losses)),train=train_score,validation=valid,
                     seconds=time.monotonic()-begin)
            history.append(rec)
            with (out/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(rec,allow_nan=False)+'\n')
            value=valid['r']['all11']
            if value>best:
                best=value;best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                torch.save(dict(model=best_state,epoch=epoch,specification=spec),out/'best.pt')
            print('TRAIN',epoch,'train_r',round(train_score['r']['all11'],6),
                  'validation_r',round(value,6),'best',round(best,6),'seconds',round(time.monotonic()-begin),flush=True)
    torch.save(dict(model=model.state_dict(),epoch=100,specification=spec),out/'last.pt')
    final,_=score(forward_all(model,vd),vd,val,scales)
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state,epoch=best_epoch,specification=spec),out/'best.pt')
    best_delta=forward_all(model,vd)
    learned,learned_arrays=score(best_delta,vd,val,scales)
    train_learned,_=score(forward_all(model,data),data,rows,scales)
    train_base,_=score(torch.zeros(len(rows),7,120),data,rows,scales)
    np.savez_compressed(out/'validation_offsets.npz',delta7_uV=best_delta.double().numpy()*np.asarray(ck['scales_uV'])[list(GENERATED)][None,:,None],
        record_ids=np.asarray([r['record_id'] for r in val]))
    per_record=[]
    for i,row in enumerate(val):
        for j,name in enumerate(LEADS):
            if bool(vd['mask'][i,j]):
                per_record.append(dict(**{k:row[k] for k in ('record_id','subject_id','target_record_id','group')},lead=name,
                    A_r=float(baseline_arrays[0][i,j]),C_r=float(learned_arrays[0][i,j]),
                    delta_r=float(learned_arrays[0][i,j]-baseline_arrays[0][i,j]),
                    A_rmse_uV=float(baseline_arrays[1][i,j]),C_rmse_uV=float(learned_arrays[1][i,j])))
    with (out/'per_record_lead.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_record[0]));writer.writeheader();writer.writerows(per_record)
    subjects=sorted({r['subject_id'] for r in val})
    subject_delta={s:float(np.mean([r['delta_r'] for r in per_record if r['subject_id']==s])) for s in subjects}
    eligible_delta=learned['r']['all11']-baseline['r']['all11']
    report=dict(completed=True,specification=spec,parameters=count,best_epoch=best_epoch,
        A=baseline,B_structural_1s=oracle,B_independent11_1s_r=b_unconstrained,
        C_best=learned,C_final=final,C_train=train_learned,A_train=train_base,
        delta_r=eligible_delta,subjects_improved=sum(v>0 for v in subject_delta.values()),
        validation_subjects=len(subjects),subject_mean_delta_r=subject_delta,
        seconds=time.monotonic()-begin,exact_statistics_max_reference_r_difference=maxdiff,
        limitations=['Single seed; checkpoint selected on this validation set, no untouched test claim.',
        'Training predictions are in-sample for frozen C5; validation subjects are disjoint.',
        'Independent11 oracle breaks limb algebra; structural7 oracle preserves it.',
        'C predicts nonoverlapping1s constants; boundary jumps remain an explicit first-probe limitation.',
        'Full120 context available offline; not a real-time causal model.'])
    write(out/'completed.json',report)
    lines=['# C5 局部均值学习实验 C','',f'参数 {count:,}，冻结原 C5 EMA；359条训练配对，90条原验证配对；100 epochs、seed42。',
        '','|组别|完整120s r|宏平均RMSE µV|1s均值误差RMS µV|','|---|---:|---:|---:|']
    for name,value in [('A 原模型',baseline),('B 结构一致真值均值替换',oracle),('C 学习修正 best',learned),('C 学习修正 final',final)]:
        lines.append(f"|{name}|{value['r']['all11']:.6f}|{value['rmse_uV']['all11']:.3f}|{value['mean_error_rms_uV']['all11']:.3f}|")
    lines+=['',f"C best相对A Δr={eligible_delta:+.6f}；best epoch={best_epoch}；{report['subjects_improved']}/{len(subjects)} 位验证受试者逐导联平均Δr为正。",
        f"B独立11导联1s真值替换 r={b_unconstrained['all11']:.6f}，仅诊断，不与保持肢体公式的C混为同一结构。",
        '', '网络：50Hz的同步I和7条预测＋绝对位置，经32通道局部CNN、每秒池化、扩张卷积残差块(d=1,2,4,8)、零初始化7通道输出头。每秒修正II和V1–V6，再派生其他肢体导联。',
        '', '训练标签：每秒真值均值−原预测均值；固定训练尺度，不对输入逐窗去均值。损失：QC/物理目标/组/导联加权Huber＋0.1完整120s相关性损失。',
        '', '结论需同时看训练与验证、三设备组和逐受试者结果；这是单seed可学习性探针，验证集选点结果尚不构成独立测试确认。零修正作为epoch0候选，若best为0表示此次训练未超过原模型。',
        '', '评分通过每秒均值和段内二阶矩精确重建完整120s的r/RMSE，非短窗r平均；原始缓存与评分最大差'+str(maxdiff)+'。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETED',eligible_delta,out,flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    c=sub.add_parser('cache');c.add_argument('--shard',type=int,required=True);c.add_argument('--shards',type=int,default=3);c.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--shard-dirs',type=Path,nargs='+',required=True);t.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='prepare':
        spec,*_=sources()
        if (HERE/'plan.json').exists():
            if json.loads((HERE/'plan.json').read_text())!=spec:
                raise ValueError('Plan exists with a different identity')
        else:
            write(HERE/'plan.json',spec)
        print(json.dumps({k:spec[k] for k in ('checkpoint_sha256','train_subjects','validation_subjects','baseline_r')},indent=2))
    elif args.command=='cache':
        if not 0<=args.shard<args.shards:
            raise ValueError('Invalid shard')
        cache_records(args)
    else:
        train_corrector(args)


if __name__=='__main__':
    main()
