"""Real GPU forward/backward and K16 smoke check before the batch32 run."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from baselines.B7.config import ModelConfig, LossConfig, SamplingConfig
    from baselines.B7.model import B7UNet
    from baselines.B7.losses import flow_loss
    from baselines.B7.flow import sample
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    for seconds in (10, 40):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = B7UNet().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        ema = {k:v.detach().clone() for k,v in model.state_dict().items()}
        anchor = torch.randn(32,1,seconds*500,device=args.device)
        batch = {'anchor':anchor,'target':torch.randn(32,7,seconds*500,device=args.device),
                 'lead_mask':torch.ones(32,7,dtype=torch.bool,device=args.device)}
        model.train()
        loss,parts = flow_loss(model,batch)
        loss.backward()
        optimizer.step()
        model.eval()
        evaluated = copy.deepcopy(model).eval()
        config = SamplingConfig(window_seconds=seconds,sample_batch_size=32,validation_batch_size=8)
        prediction = sample(evaluated,anchor[:8],[str(i) for i in range(8)],config=config)
        assert torch.isfinite(prediction).all()
        torch.cuda.synchronize()
        print(json.dumps({'seconds':seconds,'batch':32,'validation_records':8,'K':16,
                          'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                          'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'status':'PASS'}),flush=True)
        del model,evaluated,optimizer,ema,anchor,batch,loss,parts,prediction


if __name__=='__main__':
    main()
