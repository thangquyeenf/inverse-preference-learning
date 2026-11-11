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


def pair_embedding(obs_1: np.ndarray, obs_2: np.ndarray, action_1: np.ndarray, action_2: np.ndarray) -> np.ndarray:
    """Model-free embedding of a pair: per-segment mean of (obs, action), both segments concatenated, (B, 2 * (S + A))."""
    seg_1 = np.concatenate([obs_1.mean(axis=1), action_1.mean(axis=1)], axis=-1)
    seg_2 = np.concatenate([obs_2.mean(axis=1), action_2.mean(axis=1)], axis=-1)
    return np.concatenate([seg_1, seg_2], axis=-1).astype(np.float32)


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a**2).sum(1)[:, None] - 2 * a @ b.T + (b**2).sum(1)[None], 0.0)


def kmeans(x: np.ndarray, num_clusters: int, rng: np.random.Generator, iters: int = 20) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding. Returns the cluster assignment of every row of x."""
    centers = x[select_kmeanspp(x - x.mean(axis=0), num_clusters, rng)]
    for _ in range(iters):
        assign = np.argmin(_sq_dists(x, centers), axis=1)
        for c in range(num_clusters):
            members = assign == c
            if members.any():
                centers[c] = x[members].mean(axis=0)
    return np.argmin(_sq_dists(x, centers), axis=1)


def typicality(x: np.ndarray, num_neighbors: int = 20) -> np.ndarray:
    """Inverse mean distance to the num_neighbors nearest neighbors (TypiClust, Hacohen et al. 2022)."""
    n = min(num_neighbors, len(x) - 1)
    if n <= 0:
        return np.ones(len(x))
    typ = np.empty(len(x))
    for start in range(0, len(x), 1024):
        d = np.sqrt(_sq_dists(x[start : start + 1024], x))
        typ[start : start + 1024] = 1.0 / (np.sort(d, axis=1)[:, 1 : n + 1].mean(axis=1) + 1e-8)
    return typ


def select_typiclust(
    embedding: np.ndarray, labeled_embedding: np.ndarray, k: int, rng: np.random.Generator, max_clusters: int = 500
) -> np.ndarray:
    """
    TypiClust: cluster labeled + unlabeled points into |L| + k clusters, then repeatedly take the most typical
    unlabeled point of the largest cluster with the fewest labeled points.
    The cluster count is capped at max_clusters for speed; capped clusters are revisited in the same order.
    """
    x = np.concatenate([labeled_embedding, embedding], axis=0)
    x = _standardize(x)
    num_labeled = len(labeled_embedding)
    assign = kmeans(x, min(num_labeled + k, max_clusters, len(x)), rng)
    labeled_counts = np.bincount(assign[:num_labeled], minlength=assign.max() + 1).astype(np.float64)
    cand_assign = assign[num_labeled:]
    sizes = np.bincount(cand_assign, minlength=len(labeled_counts)).astype(np.float64)
    order = {c: [] for c in range(len(labeled_counts))}
    for c in range(len(labeled_counts)):
        members = np.where(cand_assign == c)[0]
        if len(members) > 0:
            typ = typicality(x[num_labeled + members])
            order[c] = members[np.argsort(-typ)].tolist()
    chosen = []
    while len(chosen) < k:
        avail = np.array([len(order[c]) > 0 for c in range(len(labeled_counts))])
        # Fewest labeled points first, ties broken by the largest remaining cluster.
        key = np.where(avail, labeled_counts * 1e6 - sizes, np.inf)
        c = int(np.argmin(key))
        chosen.append(order[c].pop(0))
        labeled_counts[c] += 1
        sizes[c] -= 1
    return np.array(chosen)


def proxy_delta_score(stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Disagreement of the proxy members on which segment is better (in the spirit of OPRIDE, Yang et al. 2026): the std
    over members of R(sigma^2) - R(sigma^1). Unlike BALD it is measured on the return difference, not on a saturating
    probability, so it does not favor near ties. Each member's difference is divided by its std over the candidates, so
    members with a larger reward scale do not dominate."""
    diff = stats["proxy_return_2"] - stats["proxy_return_1"]  # (E, B)
    return (diff / (diff.std(axis=1, keepdims=True) + 1e-8)).std(axis=0)


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
    if method == "mix_bald":
        # Half of the batch at random, the other half by BALD among the rest.
        random_part = rng.choice(num, size=k // 2, replace=False)
        scores = acquisition.bald_score(probs).copy()
        scores[random_part] = -np.inf
        return np.concatenate([random_part, select_topk(scores, k - len(random_part))])
    if method == "proxy_bald":
        return select_topk(acquisition.bald_score(stats["proxy_probs"]), k)
    if method == "proxy_density_bald":
        density = typicality(_standardize(stats["raw"]))
        return select_topk(np.log(acquisition.bald_score(stats["proxy_probs"]) + 1e-8) + eta * np.log(density), k)
    if method in ("proxy_top", "mix_proxy_top"):
        # Return-aware: ask about pairs in which the better segment has a high predicted return under the proxy,
        # so labels describe near-optimal behavior rather than ties or poor behavior.
        best = np.maximum(stats["proxy_return_1"].mean(axis=0), stats["proxy_return_2"].mean(axis=0))
        if method == "proxy_top":
            return select_topk(best, k)
        random_part = rng.choice(num, size=k // 2, replace=False)
        best[random_part] = -np.inf
        return np.concatenate([random_part, select_topk(best, k - len(random_part))])
    if method == "proxy_delta":
        return select_topk(proxy_delta_score(stats), k)
    if method == "proxy_top_delta":
        # Keep the PROXY_TOP_PREFILTER * k pairs with the highest predicted best return (the exploitation filter of
        # proxy_top), then take the k on which the members disagree most about the better segment.
        best = np.maximum(stats["proxy_return_1"].mean(axis=0), stats["proxy_return_2"].mean(axis=0))
        keep = select_topk(best, min(num, PROXY_TOP_PREFILTER * k))
        return keep[select_topk(proxy_delta_score(stats)[keep], k)]
    if method == "typiclust":
        return select_typiclust(stats["raw"], stats["raw_labeled"], k, rng)
    if method == "density_bald":
        # Information density (Settles & Craven 2008): BALD weighted by typicality in the model-free embedding.
        density = typicality(_standardize(stats["raw"]))
        return select_topk(np.log(acquisition.bald_score(probs) + 1e-8) + eta * np.log(density), k)
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
ROUND2_METHODS = ["proxy_bald", "proxy_density_bald", "typiclust", "density_bald", "mix_bald"]
ROUND3_METHODS = ["proxy_top", "mix_proxy_top"]
# Multi-round variants select in MULTIROUND[m][1] equal chunks with MULTIROUND[m][0], retraining the proxy on the labels
# revealed so far before each chunk (run_two_phase.select_multiround). IPL is still trained once on the final set.
MULTIROUND = {"proxy_top_r4": ("proxy_top", 4), "proxy_top_delta_r4": ("proxy_top_delta", 4)}
ROUND9_METHODS = ["proxy_delta", "proxy_top_delta", *MULTIROUND]
PROXY_TOP_PREFILTER = 3
NEEDS_RAW = {"typiclust", "density_bald", "proxy_density_bald"}
NEEDS_PROXY = {"proxy_bald", "proxy_density_bald", "proxy_top", "mix_proxy_top", "proxy_delta", "proxy_top_delta"}  # Scored by an explicit reward ensemble, see proxy_reward.py
NEEDS_POLICY = {"policy_bald_kcenter"}
NEEDS_FEATURES = {"bald_kcenter", "badge", "policy_bald_kcenter"}
