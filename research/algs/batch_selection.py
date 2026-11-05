"""
One-shot batch selection for the two-phase protocol: a model trained on a warm-start set of labels scores the
unlabeled pool once, and a whole batch of pairs is chosen at the same time.
Diversity-aware selectors (k-center, BADGE) avoid spending the batch on near-duplicate pairs.
"""
from typing import Dict

import numpy as np

from . import acquisition


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def select_topk(scores: np.ndarray, k: int, **kwargs) -> np.ndarray:
    return np.argsort(-scores)[:k]


def select_kcenter(scores: np.ndarray, k: int, embedding: np.ndarray, prefilter: int = 3, **kwargs) -> np.ndarray:
    """Keep the top prefilter * k candidates by score, then pick k of them with greedy farthest-first traversal."""
    candidates = np.argsort(-scores)[: min(prefilter * k, len(scores))]
    emb = _standardize(embedding[candidates])
    chosen = [0]  # Start from the highest scoring candidate.
    min_dist = np.linalg.norm(emb - emb[0], axis=1)
    for _ in range(1, min(k, len(candidates))):
        nxt = int(np.argmax(min_dist))
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(emb - emb[nxt], axis=1))
    return candidates[np.array(chosen)]


def select_kmeanspp(embedding: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (D^2 sampling), as used by BADGE."""
    first = int(np.argmax(np.linalg.norm(embedding, axis=1)))
    chosen = [first]
    min_sq = np.sum((embedding - embedding[first]) ** 2, axis=1)
    for _ in range(1, k):
        total = min_sq.sum()
        if total <= 0:
            remaining = np.setdiff1d(np.arange(len(embedding)), chosen)
            chosen.extend(rng.choice(remaining, size=k - len(chosen), replace=False).tolist())
            break
        nxt = int(rng.choice(len(embedding), p=min_sq / total))
        chosen.append(nxt)
        min_sq = np.minimum(min_sq, np.sum((embedding - embedding[nxt]) ** 2, axis=1))
    return np.array(chosen)


def badge_embedding(probs: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Gradient of the Bradley-Terry loss w.r.t. the last critic layer, using the ensemble's predicted label.
    With logit_e = w_e^T x_e (+ terms that do not depend on w_e), dL/dw_e = (p_e - y_hat) x_e.
    Averaged over ensemble members, shape (B, H).
    """
    y_hat = (probs.mean(axis=0) > 0.5).astype(np.float32)
    return ((probs - y_hat[None])[..., None] * features).mean(axis=0)


def select(method: str, stats: Dict[str, np.ndarray], k: int, rng: np.random.Generator, eta: float = 1.0) -> np.ndarray:
    """Returns k indices into the candidate set described by stats."""
    probs = stats["probs"]
    num = probs.shape[1]
    k = min(k, num)
    if method == "uniform":
        return rng.choice(num, size=k, replace=False)
    if method == "std":
        return select_topk(acquisition.std_score(probs), k)
    if method == "bald":
        return select_topk(acquisition.bald_score(probs), k)
    embedding = stats["features"].mean(axis=0)  # Ensemble-mean feature difference, (B, H)
    if method == "bald_kcenter":
        return select_kcenter(acquisition.bald_score(probs), k, embedding)
    if method == "policy_bald_kcenter":
        scores = acquisition.policy_bald_score(
            probs, log_prob_1=stats["log_prob_1"], log_prob_2=stats["log_prob_2"], eta=eta
        )
        return select_kcenter(scores, k, embedding)
    if method == "badge":
        return select_kmeanspp(badge_embedding(probs, stats["features"]), k, rng)
    raise ValueError("Unknown selection method " + method)


METHODS = ["uniform", "std", "bald", "bald_kcenter", "badge", "policy_bald_kcenter"]
NEEDS_POLICY = {"policy_bald_kcenter"}
NEEDS_FEATURES = {"bald_kcenter", "badge", "policy_bald_kcenter"}
