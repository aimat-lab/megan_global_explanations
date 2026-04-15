#!/usr/bin/env python3
"""
Standalone script to compute HDBSCAN soft cluster membership scores for new molecular embeddings.

This script demonstrates how to load the fitted HDBSCAN clusterers and centroids exported by
the vgd_concept_extraction experiment and use them to score new embeddings against each
discovered concept cluster.

=== Prerequisites ===

1. A completed experiment run that produced:
   - clusterers.pkl   — pickled dict of fitted HDBSCAN objects (one per channel)
   - centroids.json   — list of cluster centroids with channel and index info

2. New embeddings to score. These must be produced by the same MEGAN model that was used
   during the experiment (same architecture, same checkpoint). Each embedding is a 2D array
   of shape (D, K) where D is the embedding dimension and K is the number of channels.
   The experiment extracts channel-specific embeddings as embedding[:, channel_index].

3. Python packages: numpy, hdbscan (same version as the experiment, e.g. 0.8.40)

=== What this script does ===

For each cluster, it computes a soft membership probability for every input embedding using
hdbscan.prediction.membership_vector(). This function queries the fitted condensed tree to
determine how likely each point is to belong to each cluster — even for points that were
never seen during the original clustering.

The output is a score per (element, cluster) pair:
    score = 1.0 - probability
    lower score = better fit to the cluster (0.0 = perfect match)

=== Usage ===

    python compute_cluster_scores.py /path/to/experiment/results/debug

This will load clusterers.pkl and centroids.json from the given directory,
generate some random dummy embeddings for demonstration, and print the scores.

To use with real embeddings, replace the dummy data section with your own
embedding loading logic (see the clearly marked section below).
"""
import os
import sys
import json
import pickle
import typing as t

import numpy as np
from hdbscan.prediction import membership_vector


def load_clusterers(experiment_path: str) -> dict:
    """
    Load the fitted HDBSCAN clusterer objects from the experiment results.

    Args:
        experiment_path: Path to the experiment archive folder (e.g. results/debug/)

    Returns:
        Dict mapping channel_index (int) -> fitted hdbscan.HDBSCAN object.
        Each clusterer has prediction_data cached, so membership_vector() works.
    """
    clusterers_path = os.path.join(experiment_path, 'clusterers.pkl')
    if not os.path.exists(clusterers_path):
        raise FileNotFoundError(
            f'clusterers.pkl not found in {experiment_path}. '
            f'Make sure the experiment was run with a version that exports clusterers.'
        )

    with open(clusterers_path, 'rb') as f:
        channel_clusterers = pickle.load(f)

    print(f'Loaded {len(channel_clusterers)} channel clusterers from {clusterers_path}')
    for ch_idx, clusterer in channel_clusterers.items():
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        n_points = len(clusterer.labels_)
        print(f'  Channel {ch_idx}: {n_clusters} clusters, fitted on {n_points} points')

    return channel_clusterers


def load_centroids(experiment_path: str) -> t.List[dict]:
    """
    Load the cluster centroid/medoid vectors from the experiment results.

    Args:
        experiment_path: Path to the experiment archive folder

    Returns:
        List of dicts, each with keys:
            - 'index': int — unique cluster index
            - 'channel': int — which explanation channel this cluster belongs to
            - 'centroid': list[float] — the centroid/medoid embedding vector (D,)
    """
    centroids_path = os.path.join(experiment_path, 'centroids.json')
    if not os.path.exists(centroids_path):
        raise FileNotFoundError(
            f'centroids.json not found in {experiment_path}. '
            f'Make sure the experiment was run with a version that exports centroids.'
        )

    with open(centroids_path, 'r') as f:
        centroids = json.load(f)

    print(f'Loaded {len(centroids)} cluster centroids from {centroids_path}')
    for c in centroids:
        print(f'  Cluster {c["index"]} (channel {c["channel"]}): '
              f'centroid dim = {len(c["centroid"])}')

    return centroids


def compute_scores(
    embeddings: np.ndarray,
    channel_clusterers: dict,
    centroids: t.List[dict],
) -> t.Dict[int, np.ndarray]:
    """
    Compute soft HDBSCAN membership scores for a set of embeddings against all clusters.

    For each cluster, this calls membership_vector() on the channel's fitted clusterer
    to get the probability that each embedding belongs to that cluster. The score is
    1.0 - probability, so lower = better fit.

    Args:
        embeddings: (N, D, K) array of graph embeddings, where:
            N = number of elements
            D = embedding dimension
            K = number of channels
            This is the same shape as graph['graph_embedding'] in the experiment.

        channel_clusterers: Dict from load_clusterers().

        centroids: List from load_centroids().

    Returns:
        Dict mapping cluster_index (int) -> (N,) array of scores.
        Lower score = better fit to that cluster.
    """
    scores: t.Dict[int, np.ndarray] = {}

    # Group centroids by channel so we only call membership_vector once per channel
    channel_clusters: t.Dict[int, t.List[dict]] = {}
    for c in centroids:
        ch = c['channel']
        if ch not in channel_clusters:
            channel_clusters[ch] = []
        channel_clusters[ch].append(c)

    for ch_idx, clusters in channel_clusters.items():
        if ch_idx not in channel_clusterers:
            print(f'  Warning: no clusterer for channel {ch_idx}, skipping')
            continue

        clusterer = channel_clusterers[ch_idx]

        # Extract the channel-specific embeddings: (N, D)
        channel_embeddings = embeddings[:, :, ch_idx].astype(np.float64)

        # membership_vector returns (N, n_clusters_in_channel) probabilities.
        # Each row sums to <= 1.0 (the remainder is the probability of being noise).
        # Columns correspond to HDBSCAN cluster labels 0, 1, 2, ... in sorted order.
        mem_vectors = membership_vector(clusterer, channel_embeddings)

        print(f'  Channel {ch_idx}: computed membership vectors, '
              f'shape = {mem_vectors.shape}')

        # Sort clusters by their hdbscan_label (which maps to column index)
        # The experiment assigns hdbscan_label = the original HDBSCAN label (0, 1, 2, ...)
        # and membership_vector columns follow the same order.
        for cluster_info in clusters:
            cl_idx = cluster_info['index']

            # Find which column in mem_vectors corresponds to this cluster.
            # Within a channel, clusters are numbered 0, 1, 2, ... by HDBSCAN.
            # We need to figure out the column index. The clusters list for this channel
            # is ordered by their 'index' (the global cluster index), but the HDBSCAN
            # labels within the channel are 0, 1, 2, ...
            # The column order matches sorted HDBSCAN labels, which is 0, 1, 2, ...
            # So the i-th cluster in this channel (sorted by HDBSCAN label) maps to column i.
            #
            # Since we don't have hdbscan_label stored in centroids.json, we use the
            # position of this cluster among its channel's clusters (sorted by global index,
            # which preserves the HDBSCAN label order because the experiment assigns them
            # sequentially).
            col_idx = sorted(clusters, key=lambda c: c['index']).index(cluster_info)

            if col_idx >= mem_vectors.shape[1]:
                print(f'  Warning: cluster {cl_idx} column {col_idx} exceeds '
                      f'membership vector columns {mem_vectors.shape[1]}, skipping')
                continue

            # Score = 1 - probability (lower = better fit)
            scores[cl_idx] = 1.0 - mem_vectors[:, col_idx]

    return scores


def main():
    # =====================================================================
    # 1. Parse the experiment path from the command line
    # =====================================================================
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} /path/to/experiment/results/debug')
        print(f'')
        print(f'The experiment folder must contain clusterers.pkl and centroids.json')
        sys.exit(1)

    experiment_path = sys.argv[1]
    if not os.path.isdir(experiment_path):
        print(f'Error: {experiment_path} is not a directory')
        sys.exit(1)

    # =====================================================================
    # 2. Load the exported experiment artifacts
    # =====================================================================
    print('=== Loading experiment artifacts ===')
    channel_clusterers = load_clusterers(experiment_path)
    centroids = load_centroids(experiment_path)

    # =====================================================================
    # 3. Prepare embeddings to score
    #
    # *** REPLACE THIS SECTION WITH YOUR OWN EMBEDDING LOADING LOGIC ***
    #
    # In practice, you would:
    #   a) Load your MEGAN model from the checkpoint
    #   b) Run inference on your molecules to get graph embeddings
    #   c) Each embedding has shape (D, K) where D = embedding dim, K = num channels
    #   d) Stack them into an (N, D, K) array
    #
    # For demonstration, we generate random dummy embeddings with the correct shape.
    # =====================================================================
    print('\n=== Preparing embeddings ===')

    # Infer dimensions from the centroids
    embedding_dim = len(centroids[0]['centroid'])
    num_channels = max(c['channel'] for c in centroids) + 1
    num_samples = 5  # number of dummy molecules

    print(f'Embedding dimension: {embedding_dim}')
    print(f'Number of channels: {num_channels}')
    print(f'Number of samples: {num_samples}')

    # Dummy embeddings — replace this with real MEGAN model output!
    # Shape: (N, D, K) — same as graph['graph_embedding'] in the experiment
    embeddings = np.random.randn(num_samples, embedding_dim, num_channels)

    # If your model produces L2-normalized embeddings (unit sphere), normalize here:
    for ch in range(num_channels):
        norms = np.linalg.norm(embeddings[:, :, ch], axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings[:, :, ch] /= norms

    # =====================================================================
    # 4. Compute scores
    # =====================================================================
    print('\n=== Computing cluster scores ===')
    scores = compute_scores(embeddings, channel_clusterers, centroids)

    # =====================================================================
    # 5. Print results
    # =====================================================================
    print('\n=== Results ===')
    print(f'{"Sample":<10}', end='')
    for cl_idx in sorted(scores.keys()):
        ch = next(c['channel'] for c in centroids if c['index'] == cl_idx)
        print(f'  ch{ch}_cl{cl_idx:>2}', end='')
    print()
    print('-' * (10 + 12 * len(scores)))

    for i in range(num_samples):
        print(f'{i:<10}', end='')
        for cl_idx in sorted(scores.keys()):
            s = scores[cl_idx][i]
            print(f'  {s:>9.4f}', end='')
        print()

    print('\nInterpretation:')
    print('  score = 1.0 - membership_probability')
    print('  0.00 = perfect cluster member')
    print('  1.00 = definitely not in this cluster')
    print('  Values between 0 and 1 indicate partial membership')


if __name__ == '__main__':
    main()
