"""Official waveform API plus timing-only metadata; no hidden target or training cache."""
from __future__ import annotations

import csv
import math
import re
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from competition_interface import input as official_input

LIMB=('I','II','III','aVR','aVL','aVF')
BODY_COLUMNS=('1','2','9','10','11','12')
NS={'h':'urn:hl7-org:v3'}


def xml_bad_observed(path):
    """Same explicit device statement rule as training QC, restricted to visible d6."""
    root=ET.parse(path).getroot();bad=set()
    pattern=re.compile(r'信号质量差\s*[（(]\s*([^）)]*)\s*[）)]')
    for node in root.iter():
        if node.tag.rsplit('}',1)[-1]!='annotation': continue
        if not any(c.tag.rsplit('}',1)[-1]=='code' and c.get('code')=='MDC_ECG_INTERPRETATION_STATEMENT' for c in node): continue
        for child in node:
            if child.tag.rsplit('}',1)[-1]!='value': continue
            for match in pattern.finditer(''.join(child.itertext())):
                bad.update(t for t in re.split(r'[,，、/\s]+',match.group(1).strip()) if t in LIMB)
    return sorted(bad)


def read_text(path):
    for encoding in ('utf-8-sig','gb18030'):
        try: return Path(path).read_text(encoding=encoding)
        except UnicodeDecodeError: pass
    raise ValueError('Cannot decode metadata: '+str(path))


def xml_rate(path):
    root=ET.parse(path).getroot()
    rates=[]
    for series in root.findall('.//h:series',NS):
        code=series.find('h:code',NS)
        if code is None or code.get('code')!='RHYTHM': continue
        for seq in series.findall('h:component/h:sequenceSet/h:component/h:sequence',NS):
            code=seq.find('h:code',NS)
            if code is None or code.get('code') not in ('TIME_ABSOLUTE','TIME_RELATIVE'): continue
            increment=seq.find('h:value/h:increment',NS)
            if increment is None: continue
            value=float(increment.get('value'));unit=increment.get('unit','s')
            factor={'s':1.,'ms':.001,'us':.000001}.get(unit)
            if factor is None or value<=0 or not math.isfinite(value): raise ValueError('Invalid XML time increment')
            rates.append(1/(value*factor))
    if not rates: return None
    if not all(math.isclose(rates[0],rate,rel_tol=1e-8) for rate in rates): raise ValueError('Conflicting XML sampling rates')
    return rates[0]


def scale_timing(path):
    """Read only sample indices/rate; voltage conversion is exclusively official input.py."""
    rows=list(csv.reader(read_text(path).splitlines()));metadata={}
    for i,row in enumerate(rows):
        if row and row[0].strip()=='Index': break
        if row and (':' in row[0] or '：' in row[0]):
            key,value=row[0].replace('：',':',1).split(':',1);metadata[key.strip()]=value.strip()
    else: raise ValueError('Scale Index header missing')
    indices=np.asarray([int(row[0]) for row in rows[i+1:] if row and any(v.strip() for v in row)],np.int64)
    if not len(indices) or np.any(np.diff(indices)<=0): raise ValueError('Scale indices empty, duplicated or reversed')
    return float(metadata['采样率']) if '采样率' in metadata else None,indices-indices[0]


def rate_for(observed,fallback,source):
    if observed is None:
        if fallback is None: raise ValueError(f'{source}: no sampling rate metadata; set an explicit verified {source}_fs_fallback in submission.json')
        observed=float(fallback)
    if not math.isfinite(observed) or not 1<=observed<=10000: raise ValueError('Invalid source sampling rate')
    return float(observed)


def resample_valid(raw,valid,fs):
    """Resample continuous valid runs independently. Never bridge a gap or squeeze time."""
    from scipy.signal import resample_poly
    x=np.asarray(raw,np.float32);mask=np.asarray(valid,bool)&np.isfinite(x)
    if x.ndim!=2 or x.shape!=mask.shape or not x.shape[1]: raise ValueError('Invalid physical history')
    if math.isclose(fs,500,rel_tol=1e-10): return np.where(mask,x,0).astype(np.float32),mask.copy()
    ratio=Fraction(500/fs).limit_denominator(10000)
    if not math.isclose(fs*ratio.numerator/ratio.denominator,500,rel_tol=1e-8): raise ValueError('Inexact sampling conversion')
    length=math.ceil(x.shape[1]*500/fs)
    out=np.zeros((len(x),length),np.float32);available=np.zeros(out.shape,bool)
    for lead in range(len(x)):
        edges=np.diff(np.r_[False,mask[lead],False].astype(np.int8))
        for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            left=math.ceil(start*500/fs);right=min(length,math.ceil(stop*500/fs))
            if right<=left: continue
            part=resample_poly(x[lead,start:stop],ratio.numerator,ratio.denominator).astype(np.float32)
            # scipy's first output corresponds to original run start, including
            # fractional grid phase; interpolate onto the global 500Hz grid.
            coordinates=(np.arange(left,right)-start*500/fs)
            out[lead,left:right]=np.interp(coordinates,np.arange(len(part)),part)
            available[lead,left:right]=True
    return out,available


class OfficialDataset:
    def __init__(self,root,config):
        self.root=Path(root).resolve();self.config=config
        self.reader=official_input.Dataset(self.root,self.root/'userinfo.txt')
        self.scale_paths={}
        for path in sorted((self.root/'scale').glob('*.csv')):
            # Official Dataset establishes subject membership. Here path stems
            # are used only to locate timing metadata for its returned record.
            if path.stem in self.scale_paths: raise ValueError('Duplicate scale record id')
            self.scale_paths[path.stem]=path

    def __iter__(self): return self.reader.iter_samples()

    def anchor(self,person):
        candidates=[];rejected=[]
        for record in person['ele']['d12']:
            try:
                path=self.root/'Electrocardiograph/d12_test'/(record['record_id']+'.xml')
                observed=xml_rate(path)
                fs=rate_for(observed,self.config.get('d12_fs_fallback'),'d12')
                raw=np.asarray(record['leads']['I'],np.float32)[None]
                if raw.ndim!=2 or raw.shape[0]!=1 or not raw.shape[1]: raise ValueError('Current I must be a nonempty one-dimensional signal')
                if not np.isfinite(raw).all() or raw.shape[1]/fs<120-1e-8: raise ValueError('Need 120 real seconds of finite current I; padding is forbidden')
                # Resample only the first 120s; do not introduce context or target centering.
                count=round(120*fs)
                if not math.isclose(count/fs,120,abs_tol=1e-8): raise ValueError('120s cannot be represented at this rate')
                current,mask=resample_valid(raw[:,:count],np.ones_like(raw[:,:count],bool),fs)
                if current.shape!=(1,60000) or not mask.all(): raise ValueError('Current I resampling must produce exactly 60000 real samples')
                candidates.append((record['record_id'],raw.shape[1]/fs,current,fs,observed is None))
            except (ValueError,KeyError,OSError) as exc: rejected.append({'record_id':record['record_id'],'reason':str(exc)})
        if not candidates: raise ValueError('No usable 120s d12 I for '+person['subject_id']+': '+str(rejected))
        candidates.sort(key=lambda item:(-item[1],item[0]))
        record,seconds,raw,fs,fallback=candidates[0]
        return raw,{'record_id':record,'input_fs':fs,'fs_fallback_used':fallback,'original_seconds':seconds,
            'selected_seconds':[0,120],'selection':'longest_usable_then_record_id','rejected':rejected}

    def history(self,person,source):
        from baselines.B5_best_context.adapters import HistoryRecord
        records=person['ele']['d6'] if source=='ecg_machine_d6' else person['scale']
        candidates=[];rejected=[]
        for record in records:
            try:
                if source=='ecg_machine_d6':
                    path=self.root/'Electrocardiograph/d6_test'/(record['record_id']+'.xml')
                    observed=xml_rate(path);positions=None;columns=LIMB;fallback=self.config.get('d6_fs_fallback')
                else:
                    path=self.scale_paths[record['record_id']]
                    observed,positions=scale_timing(path);columns=BODY_COLUMNS;fallback=self.config.get('scale_fs_fallback')
                fs=rate_for(observed,fallback,'d6' if source=='ecg_machine_d6' else 'scale')
                values=np.asarray([record['leads'][name] for name in columns],np.float32)
                if values.ndim!=2 or len(values)!=6 or not values.shape[1]: raise ValueError('History needs six nonempty equal-length channels')
                if positions is not None:
                    if len(positions)!=values.shape[1]: raise ValueError('Scale timing and official waveform row count differ')
                    if positions[-1]/fs>self.config['max_history_seconds']: raise ValueError('Scale index span exceeds configured history limit')
                    raw=np.zeros((6,int(positions[-1])+1),np.float32);valid=np.zeros_like(raw,bool)
                    raw[:,positions]=values;valid[:,positions]=np.isfinite(values)
                else:
                    raw=values;valid=np.isfinite(values)
                    for name in xml_bad_observed(path): valid[LIMB.index(name)]=False
                if raw.shape[1]/fs>self.config['max_history_seconds']: raise ValueError('History exceeds configured limit')
                # Constant lead is unavailable; no hidden device-QC/target labels are read.
                for lead in range(6):
                    points=raw[lead,valid[lead]]
                    if points.size<2 or np.ptp(points)<=1e-8: valid[lead]=False
                converted,mask=resample_valid(raw,valid,fs)
                usable=int(mask[:2].all(axis=0).sum())
                if usable<1000: raise ValueError('Fewer than 2s valid I/II history')
                history=HistoryRecord(source,record['record_id'],converted,mask)
                candidates.append((usable,record['record_id'],history,fs,observed is None))
            except (ValueError,KeyError,OSError) as exc: rejected.append({'record_id':record['record_id'],'reason':str(exc)})
        if not candidates: return None,{'selection':'fallback','rejected':rejected,'reason':'no_usable_history'}
        candidates.sort(key=lambda item:(-item[0],item[1]))
        usable,key,history,fs,fallback=candidates[0]
        return history,{'selection':'most_valid_I_II_then_record_id','record_id':key,'input_fs':fs,
            'fs_fallback_used':fallback,'valid_I_II_seconds':usable/500,'rejected':rejected,
            'lead_order':list(LIMB),'scale_columns':list(BODY_COLUMNS) if source=='body_scale_d6' else None,
            'quality_policy':'visible_finite_nonconstant_leads_index_gaps_explicit_d6_device_statements; no_target_QC'}

    def inspect(self):
        rows=[]
        for person in self:
            rows.append({'subject_id':person['subject_id'],'userinfo_available':person['userinfo'] is not None,
                'watch':len(person['watch']),'scale':len(person['scale']),'d6':len(person['ele']['d6']),
                'd12':len(person['ele']['d12']),
                'd12_I_lengths':[len(r['leads']['I']) for r in person['ele']['d12']]})
        if not rows: raise ValueError('Official data contains no subjects')
        return {'input_format_checked':True,'model_forward_checked':False,'performance_checked':False,
                'subjects':rows,'note':'Organizer example has 4 points and is format-only; never pad it into a scored record.'}
