"""
Explicit reward-model ensemble used only to score queries (a proxy acquisition model).
IPL never learns an explicit reward, and its implicit reward r = Q - gamma * V predicts held-out preferences poorly
with few labels. A small Bradley-Terry reward ensemble trained on the same labels is a much better predictor, so its
disagreement is a more useful acquisition signal. The proxy is discarded after selection; IPL training is unchanged.
"""
from typing import Dict

import numpy as np
import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _inputs(obs: np.ndarray, action: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.concatenate([obs, action], axis=-1), dtype=torch.float32)


def train_proxy(
    labeled: Dict[str, np.ndarray],
    ensemble_size: int = 5,
    hidden_dim: int = 256,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
):
    """Trains each member on a bootstrap resample of the labeled pairs. Segment return is the sum of r(s, a)."""
    torch.manual_seed(seed)
    x_1, x_2 = _inputs(labeled["obs_1"], labeled["action_1"]), _inputs(labeled["obs_2"], labeled["action_2"])
    y = torch.as_tensor(labeled["label"], dtype=torch.float32)
    members = []
    for _ in range(ensemble_size):
        net = _mlp(x_1.shape[-1], hidden_dim)
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        bootstrap = torch.randint(len(y), (len(y),))
        for _ in range(steps):
            b = bootstrap[torch.randint(len(bootstrap), (batch_size,))]
            logits = net(x_2[b]).sum(dim=(1, 2)) - net(x_1[b]).sum(dim=(1, 2))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[b])
            optim.zero_grad()
            loss.backward()
            optim.step()
        members.append(net.eval())
    return members


@torch.no_grad()
def proxy_probs(members, pairs: Dict[str, np.ndarray], chunk_size: int = 1000) -> np.ndarray:
    """P(segment 2 preferred) under each member, shape (E, B)."""
    x_1, x_2 = _inputs(pairs["obs_1"], pairs["action_1"]), _inputs(pairs["obs_2"], pairs["action_2"])
    probs = []
    for net in members:
        chunks = [
            torch.sigmoid(net(x_2[i : i + chunk_size]).sum(dim=(1, 2)) - net(x_1[i : i + chunk_size]).sum(dim=(1, 2)))
            for i in range(0, len(x_1), chunk_size)
        ]
        probs.append(torch.cat(chunks).numpy())
    return np.stack(probs)
