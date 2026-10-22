"""Deterministic, visible-I-only history grouping and current-I template matching."""
from __future__ import annotations
import numpy as np

K = 3
TEMPERATURE = .2


def unit_shape(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def cluster_bank(beats, k=K):
    """Group by historical I only. Historical II accompanies each visible I beat."""
    beats = np.asarray(beats, np.float32)
    if beats.ndim != 3 or beats.shape[1:] != (2, 300) or not np.isfinite(beats).all():
        raise ValueError("Expected finite visible historical beats [N,2,300]")
    templates = np.zeros((k, 2, 300), np.float32)
    counts, dispersion = np.zeros(k, np.float32), np.zeros(k, np.float32)
    if len(beats) < 4:
        return templates, counts, dispersion
    x = unit_shape(beats[:, 0])
    n = min(k, max(1, len(beats) // 4))
    centers = [int(np.argmax(x @ unit_shape(np.median(beats[:, 0], axis=0))))]
    for _ in range(1, n):
        distances = 1 - (x @ x[centers].T).max(axis=1)
        if distances.max() < 1e-6:
            break
        centers.append(int(distances.argmax()))
    centroids = x[centers].copy()
    for _ in range(15):
        labels = (x @ centroids.T).argmax(axis=1)
        updated = np.stack([unit_shape(x[labels == j].mean(axis=0))
                            if np.any(labels == j) else centroids[j] for j in range(len(centroids))])
        if np.allclose(updated, centroids, atol=1e-8):
            break
        centroids = updated
    labels = (x @ centroids.T).argmax(axis=1)
    for j in range(len(centroids)):
        group = beats[labels == j]
        if len(group) == 0:
            continue
        template = np.median(group, axis=0)
        template -= np.median(template, axis=1, keepdims=True)
        templates[j] = template
        counts[j] = len(group)
        dispersion[j] = np.mean(1 - x[labels == j] @ unit_shape(template[0]))
    return templates, counts, dispersion


def current_queries(anchor):
    """Twelve independent queries from synchronous I; no current hidden II is accepted."""
    from baselines.B5_best_context.adapters import HistoryRecord
    from baselines.B5_best_context.body_features import relation_features, BEAT16_ENCODING
    anchor = np.asarray(anchor, np.float32)
    if anchor.size != 60000 or not np.isfinite(anchor).all():
        raise ValueError("Expected full finite current I")
    queries, valid = [], []
    for window in anchor.reshape(12, 5000):
        # Repeated I merely reuses the unchanged peak/template extractor; never reads II.
        raw = np.repeat(window[None], 6, axis=0)
        history = HistoryRecord("body_scale_d6", "current-I-query", raw, np.ones_like(raw, bool))
        template, _, available = relation_features(history, encoding=BEAT16_ENCODING)
        queries.append(template[0])
        valid.append(available and np.linalg.norm(template[0] - template[0].mean()) > 1e-8)
    return np.asarray(queries, np.float32), np.asarray(valid, bool)


def mix_templates(queries, query_valid, templates, counts, dispersion):
    queries, templates = np.asarray(queries), np.asarray(templates)
    if queries.shape != (12, 300) or templates.shape != (K, 2, 300):
        raise ValueError("Invalid query or history bank shape")
    active = np.asarray(counts) > 0
    if not active.any():
        return np.zeros((12, 2, 300), np.float32), np.zeros((12, K), np.float32)
    similarity = unit_shape(queries) @ unit_shape(templates[:, 0]).T
    logits = similarity / TEMPERATURE
    logits[~np.asarray(query_valid, bool)] = 0
    logits[:, ~active] = -np.inf
    prior = np.asarray(counts) / (1 + np.maximum(np.asarray(dispersion), 0))
    weights = np.exp(logits - logits.max(axis=1, keepdims=True)) * prior[None]
    weights /= weights.sum(axis=1, keepdims=True)
    mixture = np.einsum("wk,kct->wct", weights, templates)
    return mixture.astype(np.float32), weights.astype(np.float32)
