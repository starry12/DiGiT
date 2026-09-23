"""Outer-unit interfaces (digit-outer-selection-v1).

CPU reference arrays are per scope, never Python objects per candidate. The
production CUDA adapter retains its fused kernel and does not use these arrays.
Scores, expanded-edge costs and physical I/O sizes are separate concepts.
"""

from dataclasses import dataclass
import hashlib
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class CandidateUnits:
    owner: int
    occurrence_start: int
    graph_ids: np.ndarray
    costs: np.ndarray
    scores: np.ndarray

    def identity(self):
        """Diagnostic binding only; never hash candidates on the training path."""
        h = hashlib.sha256()
        h.update(np.asarray([self.owner, self.occurrence_start], dtype="<i8").tobytes())
        for array in (self.graph_ids, self.costs, self.scores):
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class SelectedUnits:
    indices: np.ndarray
    remaining: Optional[int]
    stop_reason: str


@dataclass(frozen=True)
class OccurrenceChoices:
    """Explicit per-scope diagnostic choices, bound to candidates and budget."""

    candidate_sha256: str
    budget: int
    indices: tuple

    @classmethod
    def bind(cls, candidates, budget, indices):
        return cls(candidates.identity(), int(budget), tuple(indices))


class UniformNodewisePolicy:
    policy_id = "uniform_nodewise_v1"
    budget_scope = "destination"

    @staticmethod
    def candidates(owner, start, graph_ids, num_nodes, group_size):
        ids = np.asarray(graph_ids, dtype=np.int64)
        costs = (np.where(ids < num_nodes, 1, group_size).astype(np.int64, copy=False)
                 if ids.size else ids)
        # Equal values are this policy's decision, not an assumption of selector.
        return CandidateUnits(int(owner), int(start), ids, costs, costs)


def _replay_choices(candidates, budget, choices):
    if choices.candidate_sha256 != candidates.identity() or choices.budget != budget:
        raise ValueError("choice stream candidate/budget identity mismatch")
    indices = np.asarray(choices.indices)
    if indices.size and (indices.dtype.kind not in "iu" or indices.ndim != 1):
        raise ValueError("choice indices must be a one-dimensional integer sequence")
    indices = indices.astype(np.int64)
    if (np.any(indices < 0) or np.any(indices >= candidates.graph_ids.size)
            or np.unique(indices).size != indices.size):
        raise ValueError("repeated or out-of-range occurrence choice")
    if budget == -1:
        if not np.array_equal(indices, np.arange(candidates.graph_ids.size)):
            raise ValueError("full fanout must preserve every CSC occurrence in order")
        return SelectedUnits(indices, None, "all")
    remaining = budget
    for index in indices:
        if candidates.costs[index] > remaining or candidates.scores[index] <= 0:
            raise ValueError("ineligible or over-budget occurrence choice")
        remaining -= int(candidates.costs[index])
    available = np.ones(candidates.graph_ids.size, dtype=bool)
    available[indices] = False
    if np.any(available & (candidates.costs <= remaining) & (candidates.scores > 0)):
        raise ValueError("choice stream ended before selection could stop")
    return SelectedUnits(indices, remaining, "budget" if remaining == 0 else "no_fit")


def select_units(candidates, budget, device, choices=None):
    """Sequential PPS without replacement; preserve legacy Torch RNG calls.

The trusted candidate provider supplies positive costs/scores. Runtime only
exposes uniform; this primitive is not an accepted weighted GNN policy.
"""
    if budget < -1:
        raise ValueError("budget must be non-negative or -1")
    if choices is not None:
        return _replay_choices(candidates, budget, choices)
    count = candidates.graph_ids.size
    if budget == -1:
        return SelectedUnits(np.arange(count, dtype=np.int64), None, "all")
    if budget == 0 or count == 0:
        return SelectedUnits(np.empty(0, dtype=np.int64), budget,
                             "budget" if budget == 0 else "no_fit")
    available = torch.arange(count, dtype=torch.int64, device=device)
    costs = torch.as_tensor(candidates.costs, dtype=torch.int64, device=device)
    # Reuse the tensor for the uniform profile; no added upload or RNG draw.
    scores = costs if candidates.scores is candidates.costs else torch.as_tensor(
        candidates.scores, device=device)
    selected = []
    remaining = int(budget)
    while available.numel() and remaining > 0:
        fitting = costs[available] <= remaining
        eligible = available[fitting]
        if not eligible.numel():
            break
        weights = scores[eligible].to(torch.float32)
        chosen_at = torch.multinomial(weights, 1, replacement=False)
        chosen = eligible[chosen_at]
        selected.append(int(chosen.item()))
        remaining -= int(costs[chosen].item())
        available = available[available != chosen]
    return SelectedUnits(np.asarray(selected, dtype=np.int64), remaining,
                         "budget" if remaining == 0 else "no_fit")


@dataclass
class ExpandedEdges:
    sources: list
    destinations: list
    preferred_group_rows: dict
    group_counts: np.ndarray
    node_counts: np.ndarray

    @classmethod
    def empty(cls, num_scopes):
        return cls([], [], {}, np.zeros(num_scopes, dtype=np.int64),
                   np.zeros(num_scopes, dtype=np.int64))


@dataclass
class SampledFrontier:
    """Backend-neutral result; CUDA may fuse candidate/selection/expansion."""

    graph: object
    group_counts: object
    node_counts: object
    storage_profile: str
    preferred_group_rows: object = None
