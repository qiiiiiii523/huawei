"""Stable B5 selection uses reliable actual-I input and public mean-r scoring."""
from ecg12gen.evaluate import (
    RELIABLE_INPUT_MEAN_R_SCORE_PROTOCOL as SCORE_PROTOCOL,
    correlation_score as _correlation_score,
    correlation_score_from_summaries,
    write_correlation_score,
)


def correlation_score(r1, r2):
    return _correlation_score(r1, r2, score_protocol=SCORE_PROTOCOL)


def scores_from_summaries(summaries):
    return correlation_score_from_summaries(summaries, require_reliable_input=True)


def write_score_report(output, score):
    if score.get("score_protocol") != SCORE_PROTOCOL:
        raise ValueError("New B5 reports require reliable-input mean-r protocol")
    return write_correlation_score(output, score)


def diagnostic_summary(summary):
    result = {k: v for k, v in summary.items() if "bonus" not in k and k != "competition_total_score"}
    if result.get("require_reliable_input") is not True:
        raise ValueError("New B5 selection requires actual input reliability filtering")
    result.update(score_protocol=SCORE_PROTOCOL, checkpoint_selection_metric="main_correlation")
    return result
