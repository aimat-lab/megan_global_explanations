"""
Extends the base experiment "vgd_concept_extraction". This experiment implements the concept extraction
specifically for the synth2 synthetic dataset.

Parameters are tuned to generate relatively few concept clusters by using:
- Higher MIN_CLUSTER_SIZE (requires larger clusters)
- Higher MIN_SAMPLES (more conservative clustering)

Additionally, this experiment creates a CSV file with SMILES and distances to all cluster centroids
for each sample in the dataset via the post_clustering hook.
"""
import os
import pickle
import pathlib
import typing as t

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import fcluster
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from megan_global_explanations.utils import EXPERIMENTS_PATH


PATH = pathlib.Path(__file__).parent.absolute()
ASSETS_PATH = os.path.join(EXPERIMENTS_PATH, 'assets')

# == DATASET PARAMETERS ==
# The parameters determine the details related to the dataset that should be used as the basis
# of the concept extraction

# :param VISUAL_GRAPH_DATASETS:
#       This determines the visual graph dataset to be loaded for the concept clustering. This may either
#       be an absolute string path to a visual graph dataset folder on the local system. Otherwise this
#       may also be a valid string identifier for a vgd in which case it will be downloaded from the remote
#       file share instead.
VISUAL_GRAPH_DATASET: str = 'synth2'
# :param DATASET_TYPE:
#       This has the specify the dataset type of the given dataset. This may either be "regression" or
#       "classification"
DATASET_TYPE: str = 'regression'
# :param CHANNEL_INFOS:
#       This dictionary can optionally be given to supply additional information about the individual
#       explanation channels. The key should be the index of the channel and the value should again be
#       a dictionary that contains the information for the corresponding channel.
CHANNEL_INFOS: t.Dict[int, dict] = {
    0: {
        'name': 'negative',
        'color': 'lightskyblue',
    },
    1: {
        'name': 'positive',
        'color': 'lightcoral',
    }
}

# == MODEL PARAMETERS ==
# These parameters determine the details related to the model that should be used for the
# concept extraction. For this experiment, the model should already be trained and only
# require to be loaded from the disk

# :param MODEL_PATH:
#       This has to be the absolute string path to the model checkpoint file which contains the
#       specific MEGAN model that is to be used for the concept clustering.
MODEL_PATH: str = os.path.join(ASSETS_PATH, 'models', 'synth2.ckpt')


# == CLUSTERING PARAMETERS ==
# This section determines the parameters of the concept clustering algorithm itself.
# Parameters are set to generate FEWER clusters.

# :param FIDELITY_THRESHOLD:
#       This float value determines the treshold for the channel fidelity. Only elements with a
#       fidelity higher than this will be used as possible candidates for the clustering.
#       Higher value = fewer elements eligible for clustering.
FIDELITY_THRESHOLD: float = 0.5
# :param MIN_CLUSTER_SIZE:
#       This parameter determines the min cluster size for the HDBSCAN algorithm. Essentially
#       a cluster will only be recognized as a cluster if it contains at least that many elements.
#       Higher value = fewer, larger clusters.
MIN_CLUSTER_SIZE: int = 1000
# :param MIN_SAMPLES:
#       This cluster defines the HDBSCAN behavior. Essentially it determines how conservative the
#       clustering is. Roughly speaking, a larger value here will lead to less clusters while
#       lower values tend to result in more clusters.
#       Higher value = more conservative = fewer clusters.
MIN_SAMPLES: int = 100
# :param CLUSTER_SELECTION_METHOD:
#       This string value determines the method that is used to select the clusters from the HDBSCAN
#       algorithm. 'eom' (Excess of Mass) tends to produce fewer, larger clusters compared to 'leaf'.
CLUSTER_SELECTION_METHOD: str = 'eom'
# :param CLUSTER_REPRESENTATIVE:
#       This string value determines how the representative point for each cluster is calculated.
#       'centroid' computes the mean of all cluster embeddings. 'medoid' selects the actual
#       cluster member whose total distance to all other members is minimal. The medoid is more
#       robust to outliers and is always a real data point, but is more expensive to compute.
CLUSTER_REPRESENTATIVE: str = 'medoid'
# :param CLUSTER_SCORE_METHOD:
#       Selects how ``compute_cluster_score`` scores embeddings against a cluster. This choice
#       propagates through every downstream consumer of the hook: intra/inter diagnostics,
#       per-cluster score distributions, the concept-extraction CSV, and any inference-time
#       scoring that re-uses the fitted artifacts.
#         - 'membership': 1 - HDBSCAN soft membership probability from
#           ``hdbscan.prediction.membership_vector``. Produces a true [0, 1] probability-like
#           score but is known to degenerate under ``CLUSTER_SELECTION_METHOD='leaf'``
#           (returns 0 for core members), which flattens the intra/inter separation.
#         - 'distance': pairwise distance between the embedding and the cluster's
#           representative (centroid or medoid, per ``CLUSTER_REPRESENTATIVE``) using
#           ``CLUSTERING_METRIC``. Works identically regardless of cluster-selection method
#           and is the safer default when using leaf selection.
#         - 'knn': mean of the ``CLUSTER_SCORE_KNN_K`` smallest pairwise distances from
#           the embedding to the cluster's training members, under ``CLUSTERING_METRIC``.
#           Shape-aware (tracks the cluster's actual geometry rather than its centroid),
#           smooth falloff at the boundary, and unaffected by the membership_vector /
#           leaf-selection degeneracy.
CLUSTER_SCORE_METHOD: str = 'membership'
# :param CLUSTER_SCORE_KNN_K:
#       Used only when ``CLUSTER_SCORE_METHOD == 'knn'``. Number of nearest cluster
#       members to average over for the score. Smaller values make the score more
#       sensitive to individual outlier members; larger values smooth out the signal.
CLUSTER_SCORE_KNN_K: int = 5
# :param ANALYZE_CLUSTER_HIERARCHY:
#       When True, after the per-cluster intra/inter analysis the experiment runs
#       agglomerative hierarchical clustering on the existing clusters (per channel)
#       using the full member point clouds. This uncovers super-clusters that
#       represent the same semantic motif in different local contexts, without
#       requiring any predefined SMARTS patterns. Produces cluster_hierarchy.pkl,
#       dendrogram_ch{N}.png per channel, and a log summary at three cut-heights.
#       Scoring is not affected — this is an analysis-time artefact only.
ANALYZE_CLUSTER_HIERARCHY: bool = True
# :param CLUSTER_HIERARCHY_LINKAGE:
#       Linkage method passed to ``scipy.cluster.hierarchy.linkage``. 'average' is
#       the robust default and what the inter-cluster distance computation feeds.
#       'complete' or 'single' also work with the same input. 'ward' is NOT
#       compatible because it requires Euclidean distances, whereas the
#       inter-cluster distances use ``CLUSTERING_METRIC`` (manhattan).
CLUSTER_HIERARCHY_LINKAGE: str = 'average'
# :param CLUSTER_HIERARCHY_CUT:
#       Optional fraction in (0, 1] of the per-channel max merge distance at which to
#       cut the dendrogram and form super-clusters. When set (e.g. 0.8) the SMARTS
#       overlap plot is produced a second time using those super-clusters, saved as
#       ``smarts_cluster_overlap_merged.png``, letting you check whether the chosen
#       granularity groups semantically-equivalent leaves (e.g. all NH2 clusters).
#       Set to ``None`` to skip the merged plot.
CLUSTER_HIERARCHY_CUT: t.Optional[float] = 0.9
# :param SORT_SIMILARITY:
#       This boolean flag determines whether the clusters should be sorted by their similarity.
SORT_SIMILARITY: bool = True

# == PROTOTYPE OPTIMIZATION PARAMETERS ==
# These parameters configure the process of optimizing the cluster prototype representatation

# :param OPTIMIZE_CLUSTER_PROTOTYPE:
#       This boolean flag determines whether the prototype optimization should be executed at
#       all or not. If this is False, the entire optimization routine will be skipped during the
#       cluster discovery.
OPTIMIZE_CLUSTER_PROTOTYPE: bool = False
# :param DESCRIBE_PROTOTYPE:
#       This boolean flag determines whether the prototype description should be generated.
DESCRIBE_PROTOTYPE: bool = False
# :param HYPOTHESIZE_PROTOTYPE:
#       This boolean flag determines whether the prototype hypothesis should be generated.
HYPOTHESIZE_PROTOTYPE: bool = False

# == VISUALIZATION PARAMETERS ==
# These parameters determine the details of the visualizations that will be created as part of the
# artifacts of this experiment.

# :param NUM_MEDOID_EXAMPLES:
#       The number of example molecules to show per category (closest, furthest in-cluster,
#       furthest out-of-cluster) in the per-cluster medoid molecule overview images.
#       Set to 0 to disable this visualization.
NUM_MEDOID_EXAMPLES: int = 10
# :param PLOT_UMAP:
#       This boolean flag determines whether the UMAP visualization of the graph embeddings should be
#       created or not.
PLOT_UMAP: bool = True

# == SMARTS ANALYSIS PARAMETERS ==

# :param SMARTS_PATTERNS:
#       A dictionary mapping human-readable labels to SMARTS pattern strings.
#       The post_clustering hook will check each cluster member's SMILES against
#       these patterns (binary: present/absent) and generate an overlap matrix.
SMARTS_PATTERNS: t.Dict[str, str] = {
    'OH': '[OH]',
    '=O': '[#6]=O',
    'NH2': '[NH2]',
    'PYR': '[nX2]',
}

__DEBUG__ = True

experiment = Experiment.extend(
    'vgd_concept_extraction.py',
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)


# Override the base experiment's compute_cluster_score hook. Three scoring modes are
# supported via ``CLUSTER_SCORE_METHOD`` (see the parameter docstring above):
#   - 'membership': 1 - HDBSCAN soft membership probability (density-aware)
#   - 'distance':   pairwise metric distance to the cluster representative
#   - 'knn':        mean of the K smallest distances to cluster members (shape-aware)
# Convention in all cases: lower = better fit.
@experiment.hook('compute_cluster_score', default=False, replace=True)
def compute_cluster_score(e: Experiment,
                          embeddings: np.ndarray,
                          cluster_info: dict,
                          channel_clusterers: t.Optional[dict] = None,
                          **kwargs) -> np.ndarray:
    ch_idx = cluster_info['channel_index']
    hdbscan_label = cluster_info.get('hdbscan_label')
    centroid = cluster_info['centroid']

    def _distance_score() -> np.ndarray:
        return pairwise_distances(
            embeddings, centroid.reshape(1, -1), metric=e.CLUSTERING_METRIC
        ).flatten()

    # Plain metric distance to the cluster representative — works regardless of
    # whether a fitted clusterer is around and regardless of cluster selection method.
    if e.CLUSTER_SCORE_METHOD == 'distance':
        return _distance_score()

    # Shape-aware scoring: mean distance to the k nearest cluster members.
    if e.CLUSTER_SCORE_METHOD == 'knn':
        members = cluster_info.get('embeddings')
        if members is None:
            cluster_members = kwargs.get('cluster_members')
            if cluster_members is not None and hdbscan_label is not None:
                members = cluster_members.get(ch_idx, {}).get(hdbscan_label)
        if members is None or len(members) == 0:
            return _distance_score()
        members = np.asarray(members)
        k = min(int(e.CLUSTER_SCORE_KNN_K), len(members))
        dists = pairwise_distances(embeddings, members, metric=e.CLUSTERING_METRIC)
        k_smallest = np.partition(dists, k - 1, axis=1)[:, :k]
        return k_smallest.mean(axis=1)

    # Membership-probability mode. Fall back to distance when the clusterer isn't
    # available (e.g. at inference time without the pickled clusterers loaded).
    if channel_clusterers is None or ch_idx not in channel_clusterers or hdbscan_label is None:
        return _distance_score()

    from hdbscan.prediction import membership_vector

    clusterer = channel_clusterers[ch_idx]
    mem_vectors = membership_vector(clusterer, embeddings.astype(np.float64))

    # HDBSCAN labels 0, 1, 2, ... map to columns 0, 1, 2, ... in membership_vector's output.
    cluster_col = hdbscan_label

    return 1.0 - mem_vectors[:, cluster_col]


@experiment.hook('post_clustering', default=False, replace=True)
def post_clustering(e: Experiment,
                    cluster_infos: t.List[dict],
                    graphs: t.List[dict],
                    indices: t.List[int],
                    **kwargs) -> None:
    """
    Creates a CSV file containing SMILES strings and Manhattan distances from each sample's
    graph embedding to all cluster centroids. Distances are computed using the embedding
    channel that corresponds to each cluster.
    """
    e.log('post_clustering: creating embeddings score CSV...')

    # Build per-row metadata (smiles, labels, predictions, fidelity)
    rows = []
    for graph in graphs:
        row = {
            'smiles': graph.get('graph_repr', ''),
        }
        if 'graph_labels' in graph:
            labels = graph['graph_labels']
            if isinstance(labels, list) and len(labels) == 1:
                row['label'] = labels[0]
            else:
                row['label'] = labels
        if 'graph_prediction' in graph:
            row['prediction'] = graph['graph_prediction']
        if 'graph_fidelity' in graph:
            for ch_idx, fid in enumerate(graph['graph_fidelity']):
                row[f'fidelity_ch{ch_idx}'] = fid
        rows.append(row)

    # Compute scores per cluster via the hook (batch over all graphs)
    for cluster_info in cluster_infos:
        ch_idx = cluster_info['channel_index']
        cl_idx = cluster_info['index']

        all_embeddings = np.array([
            np.array(graph['graph_embedding'])[:, ch_idx] for graph in graphs
        ])
        scores = e.apply_hook(
            'compute_cluster_score',
            embeddings=all_embeddings,
            cluster_info=cluster_info,
            channel_clusterers=kwargs.get('channel_clusterers', {}),
        )
        for i, score in enumerate(scores):
            rows[i][f'score_ch{ch_idx}_cl{cl_idx}'] = float(score)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(e.path, 'embeddings_scores.csv')
    df.to_csv(csv_path, index=False)
    e.log(f'saved embeddings scores CSV with {len(df)} rows and {len(df.columns)} columns to {csv_path}')

    # ~ SMARTS pattern overlap analysis
    # For each cluster (and unclustered elements), check how many members match each SMARTS pattern.

    if e.SMARTS_PATTERNS:
        e.log('post_clustering: computing SMARTS pattern overlap...')

        pattern_labels = list(e.SMARTS_PATTERNS.keys())
        smarts_objects = {}
        for label, smarts_str in e.SMARTS_PATTERNS.items():
            smarts_objects[label] = Chem.MolFromSmarts(smarts_str)

        # Precompute SMARTS matches for every graph (binary)
        # graph_matches[i][label] = True/False
        graph_matches: t.List[t.Dict[str, bool]] = []
        n_failed = 0
        for graph in graphs:
            smiles = graph.get('graph_repr', '')
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            matches = {}
            for label, pattern in smarts_objects.items():
                if mol is not None:
                    matches[label] = mol.HasSubstructMatch(pattern)
                else:
                    matches[label] = False
            if mol is None:
                n_failed += 1
            graph_matches.append(matches)

        if n_failed > 0:
            e.log(f'  warning: {n_failed} SMILES could not be parsed by RDKit')

        # Build a mapping: dataset_index -> position in graphs list
        index_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        def plot_smarts_overlap(
            row_specs: t.List[t.Dict[str, t.Any]],
            out_name: str,
            title: str,
        ) -> None:
            """Render a SMARTS-overlap heatmap for an arbitrary row grouping.

            ``row_specs`` is a list of ``{'label': str, 'index_tuples': [(idx, ch), ...]}``
            dicts, one per row. The function appends an 'unclustered' row whose members
            are the graphs absent from every row's ``index_tuples``.
            """
            clustered_positions: t.Set[int] = set()
            for spec in row_specs:
                for dataset_idx, _ch in spec['index_tuples']:
                    clustered_positions.add(index_to_pos[dataset_idx])

            row_labels_local = [spec['label'] for spec in row_specs] + ['unclustered']
            n_rows_local = len(row_specs) + 1
            n_cols_local = len(pattern_labels)
            counts = np.zeros((n_rows_local, n_cols_local), dtype=int)
            sizes = np.zeros(n_rows_local, dtype=int)

            for row_idx, spec in enumerate(row_specs):
                for dataset_idx, _ch in spec['index_tuples']:
                    pos = index_to_pos[dataset_idx]
                    sizes[row_idx] += 1
                    for col_idx, label in enumerate(pattern_labels):
                        if graph_matches[pos][label]:
                            counts[row_idx, col_idx] += 1

            unclustered_row = n_rows_local - 1
            for pos in range(len(graphs)):
                if pos not in clustered_positions:
                    sizes[unclustered_row] += 1
                    for col_idx, label in enumerate(pattern_labels):
                        if graph_matches[pos][label]:
                            counts[unclustered_row, col_idx] += 1

            percentages = np.zeros_like(counts, dtype=float)
            for i in range(n_rows_local):
                if sizes[i] > 0:
                    percentages[i] = counts[i] / sizes[i] * 100

            fig_sm, ax_sm = plt.subplots(
                figsize=(max(6, n_cols_local * 1.5), max(4, n_rows_local * 0.8)),
            )
            im_sm = ax_sm.imshow(percentages, cmap='coolwarm', aspect='auto', vmin=0, vmax=100)
            ax_sm.set_xticks(range(n_cols_local))
            ax_sm.set_xticklabels(pattern_labels, fontsize=9)
            ax_sm.set_yticks(range(n_rows_local))
            ax_sm.set_yticklabels(
                [f'{lbl} (n={sizes[i]})' for i, lbl in enumerate(row_labels_local)],
                fontsize=8,
            )
            ax_sm.set_title(title)
            for i in range(n_rows_local):
                for j in range(n_cols_local):
                    ax_sm.text(j, i, f'{counts[i, j]}\n({percentages[i, j]:.0f}%)',
                               ha='center', va='center', fontsize=7, color='black')
            fig_sm.colorbar(im_sm, label='Match %')
            fig_sm.tight_layout()
            fig_sm.savefig(os.path.join(e.path, out_name), dpi=150)
            plt.close(fig_sm)
            e.log(f'saved {out_name} ({n_rows_local} rows, {n_cols_local} patterns)')

        # Leaf-cluster overlap (original behavior)
        leaf_specs = [
            {
                'label': f"ch{info['channel_index']}_cl{info['index']}",
                'index_tuples': info['index_tuples'],
            }
            for info in cluster_infos
        ]
        plot_smarts_overlap(
            leaf_specs,
            out_name='smarts_cluster_overlap.png',
            title='SMARTS Pattern Overlap per Cluster',
        )

        # Merged-cluster overlap: super-clusters from the hierarchical-clustering
        # post-processing stage, cut at CLUSTER_HIERARCHY_CUT * max_merge_distance.
        cut_frac = getattr(e, 'CLUSTER_HIERARCHY_CUT', None)
        hierarchy_path = os.path.join(e.path, 'cluster_hierarchy.pkl')
        if cut_frac is not None and os.path.exists(hierarchy_path):
            with open(hierarchy_path, 'rb') as f:
                cluster_hierarchy = pickle.load(f)

            channel_to_infos: t.Dict[int, t.List[dict]] = {}
            for info in cluster_infos:
                channel_to_infos.setdefault(info['channel_index'], []).append(info)

            merged_specs: t.List[t.Dict[str, t.Any]] = []
            for ch_idx, infos in channel_to_infos.items():
                infos_sorted = sorted(infos, key=lambda i: i['index'])
                if ch_idx not in cluster_hierarchy or len(infos_sorted) < 2:
                    # Single-cluster channel — pass through unchanged
                    for info in infos_sorted:
                        merged_specs.append({
                            'label': f"ch{ch_idx}_cl{info['index']}",
                            'index_tuples': info['index_tuples'],
                        })
                    continue

                Z = cluster_hierarchy[ch_idx]
                max_d = float(Z[:, 2].max()) if len(Z) > 0 else 0.0
                t_cut = float(cut_frac) * max_d
                grouping = fcluster(Z, t=t_cut, criterion='distance')

                groups: t.Dict[int, t.List[dict]] = {}
                for leaf_idx, g in enumerate(grouping):
                    groups.setdefault(int(g), []).append(infos_sorted[leaf_idx])

                for g_id, member_infos in groups.items():
                    leaves = sorted(m['index'] for m in member_infos)
                    if len(leaves) == 1:
                        label = f"ch{ch_idx}_cl{leaves[0]}"
                    else:
                        label = f"ch{ch_idx}_[{'+'.join(f'cl{i}' for i in leaves)}]"
                    tuples = [tup for m in member_infos for tup in m['index_tuples']]
                    merged_specs.append({'label': label, 'index_tuples': tuples})

            plot_smarts_overlap(
                merged_specs,
                out_name='smarts_cluster_overlap_merged.png',
                title=f'SMARTS Pattern Overlap per Super-Cluster '
                      f'(dendrogram cut at {cut_frac:.0%} of max merge distance)',
            )

    # ~ Medoid molecule overview
    # For each cluster, create a PNG image with three rows of molecule examples:
    #   1. Molecules closest to the cluster medoid (from within the cluster)
    #   2. Molecules furthest from the medoid (from within the cluster)
    #   3. Molecules furthest from the medoid among all fidelity-passing elements
    #      in the same channel that are NOT part of this cluster

    num_examples = e.NUM_MEDOID_EXAMPLES
    if num_examples > 0:
        e.log('post_clustering: creating medoid molecule overview images...')

        MOL_IMG_SIZE = (250, 200)
        LABEL_HEIGHT = 40
        TITLE_HEIGHT = 50
        ROW_LABEL_WIDTH = 160
        CELL_PADDING = 5

        # Build dataset_index -> position-in-graphs-list mapping
        _index_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        # Try to load a readable font; fall back to PIL default
        try:
            _font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
            _font_bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
            _font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        except (IOError, OSError):
            _font = ImageFont.load_default()
            _font_bold = _font
            _font_title = _font

        def _smiles_to_image(smiles: str) -> Image.Image:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                img = Image.new('RGB', MOL_IMG_SIZE, 'white')
                d = ImageDraw.Draw(img)
                d.text((10, MOL_IMG_SIZE[1] // 2), 'Invalid SMILES', fill='red')
                return img
            return Draw.MolToImage(mol, size=MOL_IMG_SIZE)

        def _create_cell(smiles: str, dist: float) -> Image.Image:
            mol_img = _smiles_to_image(smiles)
            cell_w = MOL_IMG_SIZE[0]
            cell_h = MOL_IMG_SIZE[1] + LABEL_HEIGHT
            cell = Image.new('RGB', (cell_w, cell_h), 'white')
            cell.paste(mol_img, (0, 0))
            draw = ImageDraw.Draw(cell)
            display_smi = smiles if len(smiles) <= 28 else smiles[:25] + '...'
            draw.text((5, MOL_IMG_SIZE[1] + 2), f's={dist:.4f}', fill='black', font=_font)
            draw.text((5, MOL_IMG_SIZE[1] + 15), display_smi, fill='gray', font=_font)
            return cell

        def _create_row_label(text: str, height: int) -> Image.Image:
            label = Image.new('RGB', (ROW_LABEL_WIDTH, height), 'white')
            draw = ImageDraw.Draw(label)
            lines = text.split('\n')
            line_h = 16
            y_start = (height - len(lines) * line_h) // 2
            for i, line in enumerate(lines):
                draw.text((10, y_start + i * line_h), line, fill='black', font=_font_bold)
            return label

        for info in cluster_infos:
            ch_idx = info['channel_index']
            cl_idx = info['index']

            # Collect positions of elements that belong to this cluster
            cluster_positions: set = set()
            for dataset_idx, _ in info['index_tuples']:
                cluster_positions.add(_index_to_pos[dataset_idx])

            # Collect all fidelity-passing embeddings and positions for this channel
            eligible_positions: t.List[int] = []
            eligible_embeddings: t.List[np.ndarray] = []
            eligible_smiles: t.List[str] = []
            for pos, graph in enumerate(graphs):
                if graph['graph_fidelity'][ch_idx] <= e.FIDELITY_THRESHOLD:
                    continue
                eligible_positions.append(pos)
                eligible_embeddings.append(np.array(graph['graph_embedding'])[:, ch_idx])
                eligible_smiles.append(graph.get('graph_repr', ''))

            all_embeddings = np.array(eligible_embeddings)

            # Compute scores via the pluggable hook
            scores = e.apply_hook(
                'compute_cluster_score',
                embeddings=all_embeddings,
                cluster_info=info,
                channel_clusterers=kwargs.get('channel_clusterers', {}),
            )

            # Separate into in-cluster and out-of-cluster
            in_cluster_elements: t.List[dict] = []
            out_cluster_elements: t.List[dict] = []
            for i, pos in enumerate(eligible_positions):
                entry = {
                    'pos': pos,
                    'dist': float(scores[i]),
                    'smiles': eligible_smiles[i],
                }
                if pos in cluster_positions:
                    in_cluster_elements.append(entry)
                else:
                    out_cluster_elements.append(entry)

            # Sort: in-cluster ascending (best fit first), out-cluster descending (worst fit first)
            in_cluster_elements.sort(key=lambda x: x['dist'])
            out_cluster_elements.sort(key=lambda x: x['dist'], reverse=True)

            closest = in_cluster_elements[:num_examples]
            furthest_in = in_cluster_elements[-num_examples:][::-1]
            furthest_out = out_cluster_elements[:num_examples]

            n_in = len(in_cluster_elements)
            n_total = n_in + len(out_cluster_elements)

            categories = [
                ('Best fit\n(in cluster)', closest),
                ('Worst fit\n(in cluster)', furthest_in),
                ('Worst fit\n(out of cluster,\nsame channel)', furthest_out),
            ]

            # Assemble the composite image
            cell_w = MOL_IMG_SIZE[0]
            cell_h = MOL_IMG_SIZE[1] + LABEL_HEIGHT
            row_h = cell_h + CELL_PADDING
            total_w = ROW_LABEL_WIDTH + num_examples * (cell_w + CELL_PADDING) + CELL_PADDING
            total_h = TITLE_HEIGHT + 3 * row_h + CELL_PADDING

            canvas = Image.new('RGB', (total_w, total_h), 'white')
            draw = ImageDraw.Draw(canvas)

            title = (f'Cluster {cl_idx} (Channel {ch_idx}) | '
                     f'In-cluster: {n_in} | Channel total: {n_total}')
            draw.text((ROW_LABEL_WIDTH, 10), title, fill='black', font=_font_title)

            for row_idx, (label_text, elements) in enumerate(categories):
                y_offset = TITLE_HEIGHT + row_idx * row_h

                row_label = _create_row_label(label_text, cell_h)
                canvas.paste(row_label, (0, y_offset))

                for col_idx, elem in enumerate(elements[:num_examples]):
                    cell = _create_cell(elem['smiles'], elem['dist'])
                    x = ROW_LABEL_WIDTH + col_idx * (cell_w + CELL_PADDING)
                    canvas.paste(cell, (x, y_offset))

            # Draw separator lines between rows
            for row_idx in range(1, 3):
                y = TITLE_HEIGHT + row_idx * row_h - CELL_PADDING // 2
                draw.line(
                    [(ROW_LABEL_WIDTH, y), (total_w, y)],
                    fill='lightgray',
                    width=1,
                )

            out_path = os.path.join(e.path, f'cluster_{cl_idx}_molecule_overview.png')
            canvas.save(out_path, dpi=(150, 150))
            e.log(f'  saved medoid molecule overview for cluster {cl_idx}: {out_path}')

        e.log(f'post_clustering: created {len(cluster_infos)} medoid molecule overview images')


experiment.run_if_main()
