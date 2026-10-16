"""Use the exact B5-best full-record hybrid and QC scorer with experimental FM."""
import copy
from pathlib import Path

from baselines.B5_best import SCORE_PROTOCOL
from baselines.B5_best.bundle import bundle_slow_beta
from baselines.B5_best.common import write, state_sha
from baselines.B5_best.predict import Predictor
from baselines.B5_best.validate import evaluate as evaluate_hybrid


class ExperimentalPredictor(Predictor):
    def __init__(self, bundle, device, fm):
        super().__init__(bundle, device)
        self.fm = fm.eval()
        self.slow.requires_grad_(False).eval()


def evaluate(model, bundle, datasets, device, output, group, epoch, engine=None, verify_initial=False):
    beta = bundle_slow_beta(bundle)
    if beta != .75:
        raise ValueError("V13 control, experimental validation and inference must all use beta=0.75")
    model.eval()
    engine = ExperimentalPredictor(bundle, device, model) if engine is None else engine
    if engine.slow_beta != beta:
        raise ValueError("Evaluation engine uses a different slow coefficient")
    engine.fm = model
    engine.slow.requires_grad_(False).eval()
    audit = copy.deepcopy(bundle)
    audit["provenance"].update(experiment_group=group, experiment_epoch=epoch,
                                evaluated_fm_state_sha256=state_sha(model.state_dict()),
                                stable_reference=bool(verify_initial))
    if not verify_initial:
        audit["reference_qc_r"] = None
    score = evaluate_hybrid(audit, datasets, device, output,
                            audit["provenance"]["evaluated_fm_state_sha256"],
                            verify_reference=verify_initial, predictor=engine)
    write(Path(output) / "experiment.json", {"group": group, "epoch": epoch,
          "FM_frozen": False, "slow_predictor_frozen": True, "beta": beta,
          "score_protocol": SCORE_PROTOCOL, "selection_is_final_hybrid_r": True,
          "weight_digest_type": "FM_state_sha256", "score": score})
    return score
