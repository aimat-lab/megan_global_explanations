#!/usr/bin/env python3
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
# # Inference on a single molecule doesn't need a GPU, and building the PyG
# # native extensions (torch-scatter / -sparse / -cluster / -spline-conv) from
# # source is fragile (requires a matching CUDA toolchain + nvcc). Instead we
# # pull prebuilt CPU wheels from PyTorch's and PyG's dedicated wheel indexes.
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
End-to-end example: load a trained MEGAN model, predict on a new molecule,
and score its embedding against all concept clusters using k-NN distance.

Requires the ``megan_global_explanations`` package to be installed (for the
``Clustering`` class). Run from the project virtualenv::

    python predict_and_score.py

=== What this script does, step by step ===

1. Load a trained MEGAN model from a ``.ckpt`` checkpoint file.
2. Load a molecule-processing module (``process.py``) that converts SMILES to
   graph dicts with the same features the model was trained on.
3. Convert a SMILES string into a graph dict and run MEGAN inference to get
   the graph embedding (D, K), node importances, and edge importances.
4. Load a ``.clu`` clustering archive produced by a ``vgd_concept_extraction``
   experiment. This archive contains per-channel cluster centroids, member
   embeddings, and the agglomerative linkage hierarchy.
5. Score the molecule's per-channel embedding against every cluster using the
   chosen method (``knn`` = mean of k nearest member distances, ``distance`` =
   distance to centroid). Lower score = better fit.
6. Print everything neatly to the terminal.
"""
import os
import importlib.util
import typing as t

import numpy as np

# The following import pulls in the MEGAN model class. The molecule-processing
# object is loaded dynamically from the ``process.py`` module pointed to by
# ``PROCESSING_PATH`` below — this way the exact same feature encoding used to
# build the dataset is re-used at inference time.
from graph_attention_student.torch.megan import Megan
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
SMILES: str = "C1=CC=CC=C1CC(=O)"  # ethanol

# Scoring method: 'knn' (mean of k nearest member distances) or 'distance'
# (distance to centroid). The metric is stored inside the .clu archive.
SCORE_METHOD: str = 'knn'

# Number of nearest cluster members to average over (knn mode only).
SCORE_KNN_K: int = 5


# Directory of this script file — relative paths in the configuration block above
# are resolved against this location, so the script works regardless of the
# current working directory. Absolute paths are used as-is.
SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# === Helper functions ===============================================
# =====================================================================

def resolve_path(path: str) -> str:
    """
    Return ``path`` unchanged if it is absolute, otherwise resolve it relative
    to the directory containing this script.
    """
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def load_processing(processing_path: str):
    """
    Dynamically import the given ``process.py`` module and return its top-level
    ``processing`` object. This is how the dataset's own feature-encoding class
    (e.g. ``VgdMoleculeProcessing``) is reused at inference time.
    """
    spec = importlib.util.spec_from_file_location('_vgd_processing_module', processing_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'could not load processing module from {processing_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'processing'):
        raise AttributeError(
            f'{processing_path} does not expose a top-level `processing` object'
        )
    print(f'  [+] loaded processing object from {processing_path}')
    return module.processing


def fmt_vector(vec: np.ndarray, max_items: int = 8, decimals: int = 4) -> str:
    """
    Pretty-print a 1D numpy array showing the first few and last few values,
    with an ellipsis in the middle for long vectors.
    """
    vec = np.asarray(vec).flatten()
    if len(vec) <= max_items:
        return '[' + ', '.join(f'{v:+.{decimals}f}' for v in vec) + ']'
    head = ', '.join(f'{v:+.{decimals}f}' for v in vec[: max_items // 2])
    tail = ', '.join(f'{v:+.{decimals}f}' for v in vec[-max_items // 2:])
    return f'[{head}, ... ({len(vec) - max_items} more), {tail}]'


# =====================================================================
# === Main pipeline ==================================================
# =====================================================================

def main() -> None:
    # --- 1. Sanity-check configuration ---
    print('=' * 70)
    print('predict_and_score.py')
    print('=' * 70)

    if (MODEL_PATH.startswith('<PLACEHOLDER')
            or CLUSTERING_PATH.startswith('<PLACEHOLDER')
            or PROCESSING_PATH.startswith('<PLACEHOLDER')):
        print('\n!!! Please edit MODEL_PATH, CLUSTERING_PATH and PROCESSING_PATH '
              'at the top of this script.')
        return

    model_path = resolve_path(MODEL_PATH)
    clustering_path = resolve_path(CLUSTERING_PATH)
    processing_path = resolve_path(PROCESSING_PATH)

    # --- 2. Load the MEGAN model ---
    print('\n[1/5] Loading MEGAN model...')
    model = Megan.load(model_path)
    # Put the model in eval mode — important so dropout/batch-norm behave correctly.
    model.eval()
    print(f'      model class: {model.__class__.__name__}')
    # Number of explanation channels (K). The graph embedding has shape (D, K).
    num_channels = getattr(model, 'num_channels', None)
    print(f'      num_channels = {num_channels}')

    # --- 3. Load the processing object ---
    # The ``process.py`` module shipped with the dataset exposes a fully-configured
    # ``processing`` instance that knows how to turn SMILES strings into graph dicts
    # with exactly the same node/edge features the model was trained on.
    print('\n[2/5] Loading SMILES processing module...')
    processing = load_processing(processing_path)
    print(f'      processing class: {processing.__class__.__name__}')

    # --- 4. Convert the SMILES into a graph dict ---
    print(f'\n[3/5] Processing SMILES: {SMILES!r}')
    graph = processing.process(SMILES)
    # The graph dict contains node_attributes, edge_attributes, edge_indices, etc.
    print(f'      nodes: {len(graph["node_attributes"])}, '
          f'edges: {len(graph["edge_attributes"])}')
    print(f'      node_attr dim: {graph["node_attributes"].shape[1]}, '
          f'edge_attr dim: {graph["edge_attributes"].shape[1]}')

    # --- 5. Run model inference ---
    # ``forward_graphs`` takes a list of graph dicts and returns a list of info
    # dicts (one per input graph), each containing the model's outputs and the
    # channel-split embeddings / importances.
    print('\n[4/5] Running MEGAN inference...')
    infos = model.forward_graphs([graph], batch_size=1)
    info = infos[0]

    graph_output = np.asarray(info['graph_output'])          # (O,)
    graph_embedding = np.asarray(info['graph_embedding'])    # (D, K)
    node_importance = np.asarray(info['node_importance'])    # (V, K)
    edge_importance = np.asarray(info['edge_importance'])    # (E, K)

    print(f'      graph_output     shape = {graph_output.shape}')
    print(f'      graph_embedding  shape = {graph_embedding.shape}')
    print(f'      node_importance  shape = {node_importance.shape}')
    print(f'      edge_importance  shape = {edge_importance.shape}')

    # --- 6. Load the clustering archive ---
    print('\n[5/5] Loading clustering archive...')
    clustering = Clustering.load(clustering_path)
    for ch_idx, ch_data in clustering.channels.items():
        n_cl = len(ch_data['clusters'])
        n_emb = len(ch_data['embeddings'])
        print(f'      channel {ch_idx}: {n_cl} cluster(s), {n_emb} embeddings')

    # --- 7. Score against all clusters per channel ---
    print(f'\n=== Computing cluster scores (method={SCORE_METHOD!r}) ===')
    cluster_rows: t.List[t.Tuple[str, int, float]] = []

    for ch_idx in sorted(clustering.channels.keys()):
        channel_emb = graph_embedding[:, ch_idx]
        scores = clustering.score(channel_emb, channel=ch_idx,
                                  method=SCORE_METHOD, k=SCORE_KNN_K)
        for cl_id, score in scores.items():
            cluster_rows.append((cl_id, ch_idx, score))

    # =====================================================================
    # === Neatly print everything =======================================
    # =====================================================================
    print('\n' + '=' * 70)
    print(' RESULTS')
    print('=' * 70)

    print(f'\nInput SMILES:  {SMILES}')

    print('\n-- Model prediction (graph_output) --')
    print(f'  {fmt_vector(graph_output)}')

    print('\n-- Graph embedding (per channel) --')
    # graph_embedding has shape (D, K) — one column per explanation channel.
    for ch in range(graph_embedding.shape[1]):
        vec = graph_embedding[:, ch]
        norm = float(np.linalg.norm(vec))
        print(f'  channel {ch}  (dim={vec.shape[0]}, ||v||={norm:.4f}):')
        print(f'    {fmt_vector(vec)}')

    print('\n-- Per-node importances (per channel) --')
    for ch in range(node_importance.shape[1]):
        vec = node_importance[:, ch]
        print(f'  channel {ch}: {fmt_vector(vec)}')

    print(f'\n-- Cluster scores ({SCORE_METHOD}) --')
    print(f'  {"cluster":<16} {"score":>10}   {"interpretation"}')
    print(f'  {"-" * 16} {"-" * 10}   {"-" * 20}')
    finite = [s for _, _, s in cluster_rows if not np.isnan(s)]
    best = min(finite) if finite else 0.0
    worst = max(finite) if finite else 1.0
    span = max(worst - best, 1e-9)

    for cl_id, ch_idx, score in cluster_rows:
        if np.isnan(score):
            badge = 'no member data'
        else:
            rel = (score - best) / span
            if rel < 0.25:
                badge = 'closest cluster'
            elif rel < 0.6:
                badge = 'partial fit'
            else:
                badge = 'far from cluster'

        print(f'  {cl_id:<16} {score:>10.4f}   {badge}')

    print('\nDone.')


if __name__ == '__main__':
    main()
