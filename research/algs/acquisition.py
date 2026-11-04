"""
Query acquisition functions for active preference learning with IPL.
Each function maps ensemble preference probabilities (and optional extras) to a score per candidate pair.
Higher scores are queried first.
"""
from typing import Optional

import numpy as np

EPS = 1e-6


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def uniform_score(probs: np.ndarray, **kwargs) -> np.ndarray:
    return np.random.rand(probs.shape[1])


def std_score(probs: np.ndarray, **kwargs) -> np.ndarray:
    # The disagreement criterion originally used in IPL_IQL._collect_feedback
    return np.std(probs, axis=0)


def entropy_score(probs: np.ndarray, **kwargs) -> np.ndarray:
    return binary_entropy(probs.mean(axis=0))


def bald_score(probs: np.ndarray, **kwargs) -> np.ndarray:
    # Mutual information between the label and the ensemble member: H[mean p] - mean H[p]
    return binary_entropy(probs.mean(axis=0)) - binary_entropy(probs).mean(axis=0)


def policy_bald_score(
    probs: np.ndarray, log_prob_1: Optional[np.ndarray] = None, log_prob_2: Optional[np.ndarray] = None, eta: float = 1.0, **kwargs
) -> np.ndarray:
    """
    BALD weighted by how likely the current policy is to generate both segments.
    score = BALD * (w1 * w2)^eta with w = exp(mean_t log pi(a_t|s_t)), computed in log space.
    The policy term is z-scored over the candidates so that eta is on the same scale across tasks.
    """
    assert log_prob_1 is not None and log_prob_2 is not None
    log_w = log_prob_1 + log_prob_2
    log_w = (log_w - log_w.mean()) / (log_w.std() + EPS)
    return np.log(bald_score(probs) + EPS) + eta * log_w


ACQUISITION_FNS = {
    "uniform": uniform_score,
    "std": std_score,
    "entropy": entropy_score,
    "bald": bald_score,
    "policy_bald": policy_bald_score,
}

NEEDS_POLICY = {"policy_bald"}
