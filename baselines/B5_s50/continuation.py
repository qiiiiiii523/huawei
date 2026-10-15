"""Budget and archive bookkeeping. No model construction or optimizer updates."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_budget(base_epochs, checkpoint=None, requested=None):
    """Only extend at a completed budget boundary, retaining the saved optimizer."""
    previous = int(checkpoint.get('total_epochs', base_epochs)) if checkpoint else int(base_epochs)
    total = previous if requested is None else int(requested)
    if total <= 0 or total < previous:
        raise ValueError('Total epochs must be positive and cannot decrease the saved budget')
    if total > previous:
        if checkpoint is None:
            raise ValueError('Extending the budget requires --resume from this run last.pt')
        if int(checkpoint['epoch']) != previous:
            raise ValueError('Finish the saved budget before extending it')
    return total, previous


def evaluation_directory(run, total_epochs, base_epochs=20):
    name = 'evaluations' if int(total_epochs) == int(base_epochs) else f'evaluations_epoch{int(total_epochs)}'
    return Path(run) / name


def archive_boundary(checkpoints, epoch):
    """Keep aliases and exact resumable state; do not duplicate validation arrays."""
    checkpoints = Path(checkpoints)
    archive = checkpoints.with_name(checkpoints.name + f'_epoch{int(epoch)}')
    marker = archive / 'archive_manifest.json'
    if marker.exists():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old.get('epoch') != int(epoch) or old['files'].get('last.pt') != file_digest(checkpoints / 'last.pt'):
            raise ValueError('Boundary archive already belongs to different checkpoints')
        for name, expected in old['files'].items():
            if file_digest(archive / name) != expected:
                raise ValueError('Boundary archive changed: ' + name)
        return archive
    files = sorted(p for p in checkpoints.iterdir() if p.is_file() and p.suffix in ('.pt', '.json', '.jsonl'))
    required = {'last.pt', 'best.pt', 'best_r.pt', 'best_bonus.pt',
                'history.jsonl', 'resolved_config.json', 'initial_evaluation.json', 'completed.json'}
    if not required.issubset({p.name for p in files}):
        raise ValueError('Cannot archive an incomplete training boundary')
    done = json.loads((checkpoints / 'completed.json').read_text(encoding='utf-8-sig'))
    if not done.get('completed') or int(done.get('total_epochs', done['additional_epochs'])) != int(epoch):
        raise ValueError('Extend only a completed run; do not overwrite an unfinished stage')
    expected = {p.name: file_digest(p) for p in files}
    archive.mkdir(parents=True, exist_ok=True)
    for source in files:
        destination = archive / source.name
        if destination.exists():
            if file_digest(destination) != expected[source.name]:
                raise ValueError('Boundary archive changed: ' + source.name)
        else:
            shutil.copy2(source, destination)
            if file_digest(destination) != expected[source.name]:
                raise ValueError('Boundary archive copy failed: ' + source.name)
    marker.write_text(json.dumps({'epoch': int(epoch), 'files': expected}, indent=2) + '\n', encoding='utf-8')
    return archive


def check_implementation(saved, current, profile_path):
    """Accept exact current code or one explicitly reviewed legacy-to-current migration."""
    if saved == current:
        return
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    if saved != profile['legacy'] or current != profile['current']:
        raise ValueError('Implementation changed after prepare; not an approved continuation migration')


def completed_evaluation_matches(checkpoint, marker):
    if not Path(marker).is_file():
        return False
    value = json.loads(Path(marker).read_text(encoding='utf-8-sig'))
    return bool(value.get('completed')) and value.get('checkpoint_sha256') == file_digest(checkpoint)
