"""
Writes a copy of a comparison pool with noisy labels, for testing query selection under an imperfect teacher.

Two teacher models, both set to the same expected disagreement with the clean labels:
  flip  each label is flipped independently with probability `rate` (a "mistake" teacher; noise does not depend
        on the pair)
  bt    labels are sampled from a Bradley-Terry teacher, P(segment 2 preferred) = sigmoid(beta * dR / std(dR)),
        so near-tied pairs are noisy and clear pairs are not (closer to human labels). dR comes from an oracle reward
        model trained on 90% of the clean labels (scripts/diagnose_selection.py), and beta is set by bisection so
        that the expected disagreement with the clean labels equals `rate`.
The output keeps obs/actions unchanged, stores the noisy labels in "label" and the original ones in "label_clean".
Use it with `run_two_phase.py --pool-tag _<tag>`.

Example:
    python scripts/make_noisy_pool.py --task window-open --teacher bt --rate 0.2   # writes window-open_10000_bt20.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))


def bt_probs(dr, beta):
    return 1 / (1 + np.exp(-beta * dr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--teacher", choices=("flip", "bt"), required=True)
    parser.add_argument("--rate", type=float, default=0.2, help="Expected disagreement with the clean labels")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    src = f"datasets/metaworld/{args.task}_10000.npz"
    out = f"datasets/metaworld/{args.task}_10000_{args.teacher}{round(100 * args.rate)}.npz"
    assert not os.path.exists(out), "Refusing to overwrite " + out
    data = dict(np.load(src))
    clean = data["label"]
    rng = np.random.default_rng(args.seed)

    if args.teacher == "flip":
        noisy = np.where(rng.uniform(size=len(clean)) < args.rate, 1 - clean, clean)
    else:
        import torch
        from diagnose_selection import oracle_returns

        torch.set_num_threads(1)
        r1, r2 = oracle_returns(args.task, seed=args.seed)
        dr = (r2 - r1) / (r2 - r1).std()
        # Disagreement with the clean label: P(label 1) when clean is 0, P(label 0) when clean is 1.
        disagree = lambda beta: np.where(clean > 0.5, 1 - bt_probs(dr, beta), bt_probs(dr, beta)).mean()  # noqa: E731
        lo, hi = 1e-3, 1e3  # disagreement decreases in beta
        assert disagree(hi) < args.rate < disagree(lo), f"rate {args.rate} not reachable: [{disagree(hi):.3f}, {disagree(lo):.3f}]"
        for _ in range(100):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if disagree(mid) > args.rate else (lo, mid)
        beta = np.sqrt(lo * hi)
        noisy = (rng.uniform(size=len(clean)) < bt_probs(dr, beta)).astype(clean.dtype)
        print(f"beta {beta:.3f}, expected disagreement {disagree(beta):.3f}")

    data["label_clean"], data["label"] = clean, noisy.astype(clean.dtype)
    np.savez(out, **data)
    print(f"Saved {out}: realized disagreement {(noisy != clean).mean():.3f}")
