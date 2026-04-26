#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "rdkit",
#     "torch==2.5.0",
#     "scikit-learn",
#     "scipy",
#     "visual-graph-datasets",
#     "graph-attention-student>=1.1.0",
#     "megan-global-explanations>=0.3.0",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [[tool.uv.index]]
# name = "pyg-cpu"
# url = "https://data.pyg.org/whl/torch-2.5.0+cpu.html"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torch-scatter = { index = "pyg-cpu" }
# torch-sparse = { index = "pyg-cpu" }
# torch-cluster = { index = "pyg-cpu" }
# torch-spline-conv = { index = "pyg-cpu" }
# ///
"""
cluster_scoring.py — minimal educational example of the ``Clustering`` API.

This script is a stripped-down companion to ``predict_and_score.py``. It walks
through the same end-to-end pipeline (load model → embed a SMILES → score
against a saved ``.clu`` clustering archive) but without any of the pretty-
printing or interpretation logic that obscures how the API is actually used.

Run it directly with uv (no virtualenv setup needed)::

    uv run cluster_scoring.py

Edit the configuration block below to point at your own model checkpoint,
clustering archive, and ``process.py`` module before running.

How it works
------------
A trained MEGAN model produces, for each input graph, a *graph embedding* of
shape ``(D, K)`` — ``D``-dimensional vectors split across ``K`` explanation
channels. The ``vgd_concept_extraction`` experiment clusters these per-channel
embeddings across a whole dataset and saves the result as a ``.clu`` archive.

To score a *new* molecule against those clusters, you:

1. Run the same model on the new SMILES to obtain its ``(D, K)`` embedding.
2. Call ``model.leave_one_out_deviations`` to derive a per-channel *fidelity*
   value — how much each channel actually contributes to the prediction.
3. Load the archive with ``Clustering.load(path)``.
4. For each channel, call ``clustering.score(emb_k, channel=k, ...)`` with the
   1D embedding for that channel. The method returns a dict mapping cluster
   ids (e.g. ``"ch0_cl3"``) to scalar scores. Lower = better fit. When
   ``sharpen`` is enabled the scores live in ``[0, 1]`` where ``1.0`` means
   "not assigned"; when fidelity falls below ``fidelity_threshold`` the
   score method itself returns all-1.0 for that channel.

How to use
----------
* Set ``MODEL_PATH``, ``CLUSTERING_PATH``, ``PROCESSING_PATH`` and ``SMILES``.
* Pick a ``SCORE_METHOD`` — ``'knn'`` (mean of ``k`` nearest member distances)
  or ``'distance'`` (distance to centroid).
* Optionally enable sharpening (``'sparsemax'`` or ``'softmax'``) to turn raw
  distances into bounded ``[0, 1]`` membership-like scores.
* ``FIDELITY_THRESHOLD`` and ``DATASET_TYPE`` must match the values used when
  the clustering archive was produced.
"""
import os
import typing as t

import numpy as np

from graph_attention_student.torch.megan import Megan
from visual_graph_datasets.util import dynamic_import
from megan_global_explanations.data import Clustering


# =====================================================================
# === Configuration — edit these before running ======================
# =====================================================================

# Absolute path to the trained MEGAN model checkpoint (.ckpt file).
# Example: "/media/ssd/Programming/megan_global_explanations/megan_global_explanations/experiments/assets/models/synth2.ckpt"
MODEL_PATH: str = "model.ckpt"

# Absolute path to the ``.clu`` clustering archive produced by an earlier
# vgd_concept_extraction experiment run.
CLUSTERING_PATH: str = "clustering.clu"

# Absolute path to a ``process.py`` module that exposes a top-level ``processing``
# object (an instance of a ``ProcessingBase`` subclass — e.g. ``MoleculeProcessing``
# or a dataset-specific variant). This is the same file that ships alongside the
# visual graph dataset and guarantees feature-encoding parity with training.
PROCESSING_PATH: str = "process.py"

# A SMILES string to run prediction + cluster scoring for. The default below is
# a trivial example; replace it with any molecule of interest.
SMILES: str = "C1=CC=CC=C1CC(=O)NC"

# Scoring method: 'knn' (mean of k nearest member distances) or 'distance'
# (distance to centroid). The metric is stored inside the .clu archive.
SCORE_METHOD: str = 'knn'

# Number of nearest cluster members to average over (knn mode only).
SCORE_KNN_K: int = 5

# Optional post-processing to force more decisive cluster assignments.
#   None         — return raw distances
#   'sparsemax'  — produces exact zeros for weak matches when one cluster clearly wins
#   'softmax'    — smoother, never exactly zero
SCORE_SHARPEN: t.Optional[str] = 'sparsemax'

# Temperature multiplier for the sharpening (lower = more discrete).
SCORE_SHARPEN_TEMPERATURE: float = 0.5

# Per-channel fidelity threshold. Channels where the molecule's fidelity falls
# below this are treated as "not assigned to any cluster" — sharpening is
# skipped and all cluster scores in that channel are set to 1.0. Match the
# FIDELITY_THRESHOLD used when the experiment was run.
FIDELITY_THRESHOLD: float = 0.1

# Dataset type — determines how per-channel fidelity is derived from the
# model's leave-one-out deviations. Must match the experiment that produced
# the Clustering. Either 'regression' or 'classification'.
DATASET_TYPE: str = 'regression'


# Resolve relative paths against this script's directory so the script works
# regardless of the current working directory.
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path: str) -> str:
    """Return ``path`` unchanged if absolute, otherwise relative to SCRIPT_DIR."""
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def compute_graph_fidelity(deviation: np.ndarray, dataset_type: str) -> np.ndarray:
    """Derive a per-channel fidelity vector from MEGAN's leave-one-out deviation.

    Mirrors the formula used inside ``vgd_concept_extraction.py`` so the scale
    matches whatever ``FIDELITY_THRESHOLD`` was calibrated against at training.
    """
    dev = np.asarray(deviation)
    if dataset_type == 'regression':
        return np.array([-dev[0, 0], dev[0, 1]])
    if dataset_type == 'classification':
        return np.diag(dev)
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


def main() -> None:
    # 1. Resolve config paths.
    model_path = resolve_path(MODEL_PATH)
    clustering_path = resolve_path(CLUSTERING_PATH)
    processing_path = resolve_path(PROCESSING_PATH)

    # 2. Load the trained MEGAN model and put it in eval mode.
    model = Megan.load(model_path)
    model.eval()

    # 3. Load the dataset's processing object — this is what turns a SMILES
    #    string into a graph dict with the exact features the model expects.
    #    ``dynamic_import`` from visual_graph_datasets imports the file at the
    #    given path as a module; ``process.py`` exposes a top-level ``processing``.
    processing = dynamic_import(processing_path, name='vgd_processing_module').processing

    # 4. Convert SMILES → graph dict and run the model on it. ``forward_graphs``
    #    accepts a list of graph dicts and returns a list of info dicts with
    #    ``graph_output``, ``graph_embedding`` (D, K), ``node_importance``, etc.
    graph = processing.process(SMILES)
    info = model.forward_graphs([graph], batch_size=1)[0]
    graph_embedding = np.asarray(info['graph_embedding'])  # (D, K)

    # 5. Per-channel fidelity. ``leave_one_out_deviations`` masks one channel
    #    at a time and measures how the prediction shifts; the formula in
    #    ``compute_graph_fidelity`` reduces that to one scalar per channel.
    deviations = model.leave_one_out_deviations([graph])
    graph_fidelity = compute_graph_fidelity(deviations[0], DATASET_TYPE)

    # 6. Load the saved clustering archive. Everything needed for scoring
    #    (per-channel centroids, member embeddings, metric, hierarchy) lives
    #    inside the .clu file — no dataset or model state required.
    clustering = Clustering.load(clustering_path)

    # 7. Score each channel separately. ``Clustering.score`` takes a 1D
    #    embedding vector and returns ``{cluster_id: score}``. Passing
    #    ``fidelity`` and ``fidelity_threshold`` lets the method short-circuit
    #    low-fidelity channels to all-1.0 when sharpening is enabled.
    scores: t.Dict[str, float] = {}
    for ch_idx in sorted(clustering.channels.keys()):
        channel_emb = graph_embedding[:, ch_idx]
        channel_scores = clustering.score(
            channel_emb,
            channel=ch_idx,
            method=SCORE_METHOD,
            k=SCORE_KNN_K,
            sharpen=SCORE_SHARPEN,
            sharpen_temperature=SCORE_SHARPEN_TEMPERATURE,
            fidelity=float(graph_fidelity[ch_idx]),
            fidelity_threshold=FIDELITY_THRESHOLD,
        )
        scores.update(channel_scores)

    # 8. Single result print.
    print({cl_id: round(s, 4) for cl_id, s in scores.items()})


if __name__ == '__main__':
    main()
