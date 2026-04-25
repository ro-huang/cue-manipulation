"""CAA steering-vector computation.

Algorithm (Rimsky et al. / Panickssery et al. "Contrastive Activation Addition"):
  1. For each contrast pair, run prompt + confident_completion through the model and
     capture the hidden state at the post-newline position at layer L.
  2. Do the same for prompt + hedged_completion.
  3. Steering vector v_L = mean(h_confident - h_hedged) over pairs.
  4. Confidence projection of a test trial h_test is dot(h_test, v_L) / ||v_L||.

Only `project` / `load_vector` / `save_vector` are torch-free; `build_caa_vector`
imports torch lazily so importing this module stays cheap when only projection is
needed (e.g., at analysis time).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .contrast_pairs import ContrastPair

if TYPE_CHECKING:  # pragma: no cover
    from ..models import HFModel


def _post_newline_position(tokenizer, prompt: str, completion: str) -> int:
    """Return the absolute token index (in the full sequence) of the final '\n' in the
    completion. Fallback: last token of the sequence.
    """
    full = prompt + completion
    full_ids = tokenizer(full, return_tensors="pt")["input_ids"][0].tolist()
    for i in range(len(full_ids) - 1, -1, -1):
        s = tokenizer.decode([full_ids[i]])
        if "\n" in s:
            return i
    return len(full_ids) - 1


def build_caa_vector(
    model: "HFModel",
    pairs: list[ContrastPair],
    layer: int | None = None,
) -> dict:
    """Compute the CAA steering vector from a list of contrast pairs."""
    import torch  # noqa: F401
    from tqdm import tqdm

    layer_idx = layer if layer is not None else model.caa_layer
    resolved = (
        layer_idx
        if layer_idx >= 0
        else model.model.config.num_hidden_layers + 1 + layer_idx
    )

    diffs = []
    for pair in tqdm(pairs, desc="caa_pairs"):
        conf_pos = _post_newline_position(model.tokenizer, pair.prompt, pair.confident)
        hedg_pos = _post_newline_position(model.tokenizer, pair.prompt, pair.hedged)
        conf_hs = model.hidden_states_at(
            pair.prompt + pair.confident, positions=[conf_pos], layers=[resolved]
        )[(resolved, conf_pos)]
        hedg_hs = model.hidden_states_at(
            pair.prompt + pair.hedged, positions=[hedg_pos], layers=[resolved]
        )[(resolved, hedg_pos)]
        diffs.append(conf_hs - hedg_hs)

    import torch
    stacked = torch.stack(diffs, dim=0)
    v = stacked.mean(dim=0).numpy()
    return {
        "vector": v.astype(np.float32),
        "layer": resolved,
        "n_pairs": len(pairs),
        "norm": float(np.linalg.norm(v)),
    }


def save_vector(path: Path, data: dict) -> None:
    np.savez(path, **data)


def load_vector(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k].item() if npz[k].ndim == 0 else npz[k] for k in npz.files}


def project(hidden: np.ndarray, vector: np.ndarray) -> float:
    """Return dot(hidden, vector) / ||vector||."""
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return 0.0
    return float(np.dot(hidden, vector) / norm)
