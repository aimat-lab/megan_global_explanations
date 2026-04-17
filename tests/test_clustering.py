"""Tests for the Clustering archive class in megan_global_explanations.data."""
import os
import logging
import tempfile
import typing as t

import numpy as np
import pytest

from megan_global_explanations.data import Clustering, ClusterView, select_representatives


# ── Helpers ──────────────────────────────────────────────────────────

def make_synthetic_clustering(
    num_channels: int = 2,
    clusters_per_channel: int = 3,
    members_per_cluster: int = 50,
    embedding_dim: int = 16,
    metric: str = 'manhattan',
    include_representatives: bool = False,
) -> Clustering:
    """Build a minimal Clustering from random data for test purposes."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import pairwise_distances

    obj = Clustering()
    obj.embedding_dim = embedding_dim

    global_idx = 0
    for ch in range(num_channels):
        K = clusters_per_channel
        all_embeddings_parts = []
        all_labels_parts = []
        clusters = []

        for k in range(K):
            center = np.random.randn(embedding_dim).astype(np.float64)
            members = center + 0.1 * np.random.randn(members_per_cluster, embedding_dim)
            members = members.astype(np.float64)
            centroid = members.mean(axis=0)

            cl_dict = {
                'index': global_idx,
                'hdbscan_label': k,
                'centroid': centroid,
                'members': members,
                'metadata': {'name': f'cluster_{global_idx}', 'color': 'gray'},
                'annotations': {},
            }

            if include_representatives:
                n_nodes = np.random.randint(5, 15)
                n_edges = np.random.randint(4, 20)
                n_reps = min(4, members_per_cluster)
                dists = pairwise_distances(
                    members, centroid.reshape(1, -1), metric=metric
                ).flatten()
                rep_idx = np.argsort(dists)[:n_reps]
                reps = []
                for ri in rep_idx:
                    reps.append({
                        'smiles': f'C{"C" * np.random.randint(1, 5)}O',
                        'dataset_index': int(ri),
                        'node_importances': np.random.rand(n_nodes, num_channels).tolist(),
                        'edge_importances': np.random.rand(n_edges, num_channels).tolist(),
                        'graph_output': [float(np.random.rand())],
                        'graph_deviation': np.random.rand(1, num_channels).tolist(),
                    })
                cl_dict['representatives'] = reps
                cl_dict['member_stats'] = {
                    'graph_outputs': np.random.rand(members_per_cluster),
                    'graph_deviations': np.random.rand(members_per_cluster, num_channels),
                    'mask_sizes': np.random.rand(members_per_cluster, num_channels),
                    'centroid_distances': dists,
                }

            clusters.append(cl_dict)

            all_embeddings_parts.append(members)
            all_labels_parts.append(np.full(members_per_cluster, k, dtype=int))
            global_idx += 1

        # Add some noise points
        noise = np.random.randn(10, embedding_dim).astype(np.float64)
        all_embeddings_parts.append(noise)
        all_labels_parts.append(np.full(10, -1, dtype=int))

        all_embeddings = np.concatenate(all_embeddings_parts)
        all_labels = np.concatenate(all_labels_parts)

        # Build a linkage matrix if >=2 clusters
        linkage_mat = None
        if K >= 2:
            inter = np.zeros((K, K))
            for i in range(K):
                for j in range(i + 1, K):
                    d = pairwise_distances(
                        clusters[i]['members'], clusters[j]['members'], metric=metric
                    ).mean()
                    inter[i, j] = d
                    inter[j, i] = d
            linkage_mat = linkage(squareform(inter, checks=False), method='average')

        obj.channels[ch] = {
            'name': f'channel_{ch}',
            'color': 'blue' if ch == 0 else 'red',
            'metric': metric,
            'embeddings': all_embeddings,
            'labels': all_labels,
            'linkage': linkage_mat,
            'clusters': clusters,
        }

    obj._build_index()
    return obj


# ── ClusterView ──────────────────────────────────────────────────────

class TestClusterView:

    def test_getitem(self):
        view = ClusterView({'id': 'ch0_cl0', 'centroid': np.zeros(4)})
        assert view['id'] == 'ch0_cl0'

    def test_getattr(self):
        view = ClusterView({'id': 'ch0_cl0', 'centroid': np.zeros(4)})
        assert view.id == 'ch0_cl0'
        assert view.centroid.shape == (4,)

    def test_missing_attr_raises(self):
        view = ClusterView({'id': 'ch0_cl0'})
        with pytest.raises(AttributeError):
            _ = view.nonexistent

    def test_contains(self):
        view = ClusterView({'id': 'ch0_cl0', 'centroid': np.zeros(4)})
        assert 'centroid' in view
        assert 'missing' not in view

    def test_keys_values_items(self):
        data = {'id': 'ch0_cl0', 'val': 42}
        view = ClusterView(data)
        assert set(view.keys()) == {'id', 'val'}
        assert 42 in list(view.values())
        assert ('val', 42) in list(view.items())

    def test_repr(self):
        view = ClusterView({'id': 'ch0_cl0'})
        assert 'ch0_cl0' in repr(view)


# ── Clustering basics ────────────────────────────────────────────────

class TestClusteringBasics:

    def test_construction(self):
        c = make_synthetic_clustering(num_channels=2, clusters_per_channel=3)
        assert c.embedding_dim == 16
        assert len(c.channels) == 2
        assert len(c) == 6  # 2 channels * 3 clusters

    def test_getitem(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)
        cl = c['ch0_cl0']
        assert isinstance(cl, ClusterView)
        assert cl.centroid.shape == (16,)
        assert cl.members.shape[1] == 16

    def test_getitem_missing_raises(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=1)
        with pytest.raises(KeyError):
            _ = c['ch99_cl99']

    def test_iter(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        ids = list(c)
        assert len(ids) == 3
        assert all(isinstance(i, str) for i in ids)

    def test_contains(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)
        assert 'ch0_cl0' in c
        assert 'ch99_cl0' not in c


# ── Save / Load ──────────────────────────────────────────────────────

class TestClusteringSaveLoad:

    def test_round_trip(self):
        original = make_synthetic_clustering(num_channels=2, clusters_per_channel=3)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            original.save(path)

            assert os.path.exists(path)

            loaded = Clustering.load(path)

        assert loaded.schema_version == original.schema_version
        assert loaded.embedding_dim == original.embedding_dim
        assert set(loaded.channels.keys()) == set(original.channels.keys())
        assert len(loaded) == len(original)

        for ch_idx in original.channels:
            orig_ch = original.channels[ch_idx]
            load_ch = loaded.channels[ch_idx]
            assert load_ch['name'] == orig_ch['name']
            assert load_ch['color'] == orig_ch['color']
            assert load_ch['metric'] == orig_ch['metric']
            np.testing.assert_array_almost_equal(load_ch['embeddings'], orig_ch['embeddings'])
            np.testing.assert_array_equal(load_ch['labels'], orig_ch['labels'])
            assert len(load_ch['clusters']) == len(orig_ch['clusters'])

            for orig_cl, load_cl in zip(orig_ch['clusters'], load_ch['clusters']):
                assert load_cl['index'] == orig_cl['index']
                assert load_cl['hdbscan_label'] == orig_cl['hdbscan_label']
                np.testing.assert_array_almost_equal(load_cl['centroid'], orig_cl['centroid'])
                np.testing.assert_array_almost_equal(load_cl['members'], orig_cl['members'])

            if orig_ch['linkage'] is not None:
                np.testing.assert_array_almost_equal(load_ch['linkage'], orig_ch['linkage'])
            else:
                assert load_ch['linkage'] is None

    def test_round_trip_cluster_access(self):
        """Verify cluster access via [] works identically after load."""
        original = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            original.save(path)
            loaded = Clustering.load(path)

        for cl_id in original:
            orig_cl = original[cl_id]
            load_cl = loaded[cl_id]
            assert orig_cl.id == load_cl.id
            np.testing.assert_array_almost_equal(orig_cl.centroid, load_cl.centroid)

    def test_max_members_subsampling(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, members_per_cluster=200,
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path, max_members=50)
            loaded = Clustering.load(path)

        for cl_id in loaded:
            assert loaded[cl_id].members.shape[0] <= 50

    def test_max_members_does_not_affect_in_memory(self):
        """Subsampling at save time must not mutate the original object."""
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=1, members_per_cluster=100,
        )
        original_size = c[list(c)[0]].members.shape[0]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path, max_members=10)

        assert c[list(c)[0]].members.shape[0] == original_size

    def test_no_linkage_round_trip(self):
        """Channels with <2 clusters have no linkage; verify this survives save/load."""
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=1)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        assert loaded.channels[0]['linkage'] is None


# ── Scoring ──────────────────────────────────────────────────────────

class TestClusteringScore:

    def test_score_knn_returns_all_clusters(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        emb = np.random.randn(16)
        scores = c.score(emb, channel=0, method='knn', k=5)
        assert len(scores) == 3
        assert all(isinstance(v, float) for v in scores.values())
        assert all(isinstance(k, str) for k in scores.keys())

    def test_score_distance(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)
        emb = np.random.randn(16)
        scores = c.score(emb, channel=0, method='distance')
        assert len(scores) == 2

    def test_score_unknown_method_raises(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=1)
        with pytest.raises(ValueError, match='Unknown scoring method'):
            c.score(np.zeros(16), channel=0, method='bogus')

    def test_score_nearest_cluster_is_own(self):
        """A point at a cluster's centroid should score best for that cluster."""
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=3, members_per_cluster=100,
        )
        # Pick a cluster and use its centroid as the query
        target_id = list(c)[0]
        centroid = c[target_id].centroid
        scores = c.score(centroid, channel=0, method='knn', k=5)
        best_id = min(scores, key=scores.get)
        assert best_id == target_id

    def test_score_survives_save_load(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)
        emb = np.random.randn(16)
        original_scores = c.score(emb, channel=0, method='knn', k=5)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        loaded_scores = loaded.score(emb, channel=0, method='knn', k=5)
        for cl_id in original_scores:
            assert abs(original_scores[cl_id] - loaded_scores[cl_id]) < 1e-6


# ── at_linkage ───────────────────────────────────────────────────────

class TestAtLinkage:

    def test_returns_clustering(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=4)
        merged = c.at_linkage(90)
        assert isinstance(merged, Clustering)

    def test_shares_channel_data(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        merged = c.at_linkage(50)
        assert merged.channels is c.channels

    def test_fewer_or_equal_clusters(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=4)
        merged = c.at_linkage(90)
        assert len(merged) <= len(c)

    def test_at_0_percent_is_all_leaves(self):
        """At 0% cut, everything splits — each leaf is its own super-cluster."""
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        merged = c.at_linkage(0)
        # fcluster with t=0 puts everything in its own group
        assert len(merged) == len(c)

    def test_at_100_percent_merges_all(self):
        """At 100% cut, all clusters in a channel merge into one super-cluster."""
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        merged = c.at_linkage(100)
        assert len(merged) == 1

    def test_merged_cluster_has_leaves(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        merged = c.at_linkage(100)
        super_id = list(merged)[0]
        super_cl = merged[super_id]
        assert 'leaves' in super_cl
        assert len(super_cl.leaves) == 3

    def test_merged_score_uses_min(self):
        """Score on a merged view should be min of constituent leaf scores."""
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        merged = c.at_linkage(100)

        emb = np.random.randn(16)
        leaf_scores = c.score(emb, channel=0, method='knn', k=5)
        merged_scores = merged.score(emb, channel=0, method='knn', k=5)

        super_id = list(merged)[0]
        expected_min = min(leaf_scores.values())
        assert abs(merged_scores[super_id] - expected_min) < 1e-6

    def test_single_leaf_keeps_original_id(self):
        """A leaf that doesn't merge with anything should keep its original ID."""
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=1)
        merged = c.at_linkage(50)
        assert list(merged)[0] == list(c)[0]


# ── get_tree / get_forest ────────────────────────────────────────────

class TestGetTree:

    def test_returns_digraph(self):
        import networkx as nx

        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        tree = c.get_tree(channel=0)
        assert isinstance(tree, nx.DiGraph)

    def test_leaf_count(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=4)
        tree = c.get_tree(channel=0)
        leaves = [n for n, d in tree.nodes(data=True) if d.get('type') == 'leaf']
        assert len(leaves) == 4

    def test_merge_node_count(self):
        """With K leaves the linkage produces exactly K-1 merge nodes."""
        K = 5
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=K)
        tree = c.get_tree(channel=0)
        merges = [n for n, d in tree.nodes(data=True) if d.get('type') == 'merge']
        assert len(merges) == K - 1

    def test_edges_have_weight(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=3)
        tree = c.get_tree(channel=0)
        for u, v, data in tree.edges(data=True):
            assert 'weight' in data
            assert data['weight'] >= 0

    def test_leaf_attributes(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=2)
        tree = c.get_tree(channel=0)
        for n, d in tree.nodes(data=True):
            if d.get('type') == 'leaf':
                assert 'centroid' in d
                assert 'size' in d
                assert 'channel' in d

    def test_single_cluster_no_merges(self):
        c = make_synthetic_clustering(num_channels=1, clusters_per_channel=1)
        tree = c.get_tree(channel=0)
        assert len(tree.nodes) == 1
        assert len(tree.edges) == 0


class TestGetForest:

    def test_combines_channels(self):
        import networkx as nx

        c = make_synthetic_clustering(num_channels=2, clusters_per_channel=3)
        forest = c.get_forest()
        assert isinstance(forest, nx.DiGraph)

        # 2 channels * (3 leaves + 2 merge nodes) = 10 nodes
        leaves = [n for n, d in forest.nodes(data=True) if d.get('type') == 'leaf']
        merges = [n for n, d in forest.nodes(data=True) if d.get('type') == 'merge']
        assert len(leaves) == 6
        assert len(merges) == 4

    def test_disconnected_components(self):
        import networkx as nx

        c = make_synthetic_clustering(num_channels=2, clusters_per_channel=2)
        forest = c.get_forest()
        undirected = forest.to_undirected()
        components = list(nx.connected_components(undirected))
        assert len(components) == 2


# ── from_experiment ──────────────────────────────────────────────────

class TestFromExperiment:

    def test_construction(self):
        """Verify from_experiment produces a valid Clustering with correct structure."""
        D = 8
        cluster_infos = [
            {
                'channel_index': 0, 'index': 0, 'hdbscan_label': 0,
                'centroid': np.zeros(D), 'embeddings': np.random.randn(20, D),
                'name': 'neg_cl0', 'color': 'blue',
            },
            {
                'channel_index': 0, 'index': 1, 'hdbscan_label': 1,
                'centroid': np.ones(D), 'embeddings': np.random.randn(30, D),
                'name': 'neg_cl1', 'color': 'blue',
            },
            {
                'channel_index': 1, 'index': 2, 'hdbscan_label': 0,
                'centroid': np.ones(D) * 0.5, 'embeddings': np.random.randn(25, D),
                'name': 'pos_cl0', 'color': 'red',
            },
        ]
        channel_infos = {0: {'name': 'neg', 'color': 'blue'}, 1: {'name': 'pos', 'color': 'red'}}
        channel_embeddings_map = {
            0: np.random.randn(100, D),
            1: np.random.randn(80, D),
        }
        channel_labels = {
            0: np.random.choice([-1, 0, 1], size=100),
            1: np.random.choice([-1, 0], size=80),
        }

        c = Clustering.from_experiment(
            cluster_infos=cluster_infos,
            channel_infos=channel_infos,
            clustering_metric='manhattan',
            channel_embeddings_map=channel_embeddings_map,
            channel_labels=channel_labels,
        )

        assert c.embedding_dim == D
        assert len(c.channels) == 2
        assert len(c) == 3
        assert 'ch0_cl0' in c
        assert 'ch1_cl2' in c
        assert c['ch0_cl0'].centroid.shape == (D,)


# ── select_representatives ───────────────────────────────────────────

class TestSelectRepresentatives:

    def test_closest_returns_n_smallest(self):
        distances = np.array([5.0, 1.0, 3.0, 0.5, 4.0])
        indices = select_representatives(distances, n=3, strategy='closest')
        assert len(indices) == 3
        # Should be indices 3 (0.5), 1 (1.0), 2 (3.0) in sorted order
        assert set(indices) == {3, 1, 2}

    def test_closest_caps_at_length(self):
        distances = np.array([1.0, 2.0])
        indices = select_representatives(distances, n=10, strategy='closest')
        assert len(indices) == 2

    def test_closest_empty_input(self):
        distances = np.array([])
        indices = select_representatives(distances, n=5, strategy='closest')
        assert len(indices) == 0

    def test_temperature_returns_correct_count(self):
        np.random.seed(42)
        distances = np.random.rand(100)
        indices = select_representatives(distances, n=10, strategy='temperature', temperature=0.5)
        assert len(indices) == 10
        assert len(set(indices)) == 10  # no duplicates

    def test_temperature_low_biases_toward_closest(self):
        """With very low temperature, selected indices should be mostly near the closest."""
        np.random.seed(42)
        distances = np.arange(100, dtype=float)  # 0.0, 1.0, ..., 99.0
        # Very low temperature — should strongly prefer the smallest distances
        indices = select_representatives(distances, n=10, strategy='temperature', temperature=0.01)
        # All selected should be from the bottom half
        assert all(i < 50 for i in indices)

    def test_temperature_high_gives_diversity(self):
        """With very high temperature, selection should spread across the range."""
        np.random.seed(42)
        distances = np.arange(100, dtype=float)
        # Run multiple times and check we get diverse samples
        all_indices = set()
        for seed in range(20):
            np.random.seed(seed)
            indices = select_representatives(distances, n=10, strategy='temperature', temperature=100.0)
            all_indices.update(indices)
        # With high temp over 20 runs of 10 picks from 100, we should see broad coverage
        assert len(all_indices) > 50

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match='Unknown representative strategy'):
            select_representatives(np.array([1.0, 2.0]), n=1, strategy='bogus')


# ── Representatives + member_stats save/load ─────────────────────────

class TestRepresentativesSaveLoad:

    def test_round_trip_with_representatives(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, include_representatives=True,
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        for cl_id in c:
            orig_cl = c[cl_id]
            load_cl = loaded[cl_id]

            # Representatives
            assert 'representatives' in load_cl
            assert len(load_cl.representatives) == len(orig_cl.representatives)
            for orig_rep, load_rep in zip(orig_cl.representatives, load_cl.representatives):
                assert load_rep['smiles'] == orig_rep['smiles']
                assert load_rep['dataset_index'] == orig_rep['dataset_index']
                np.testing.assert_array_almost_equal(
                    load_rep['node_importances'], orig_rep['node_importances']
                )
                np.testing.assert_array_almost_equal(
                    load_rep['graph_output'], orig_rep['graph_output']
                )

    def test_round_trip_with_member_stats(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, include_representatives=True,
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        for cl_id in c:
            orig_ms = c[cl_id].member_stats
            load_ms = loaded[cl_id].member_stats

            for key in ('graph_outputs', 'graph_deviations', 'mask_sizes', 'centroid_distances'):
                np.testing.assert_array_almost_equal(load_ms[key], orig_ms[key])

    def test_without_representatives_still_works(self):
        """Clustering without representatives should save/load without error."""
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, include_representatives=False,
        )

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        assert len(loaded) == len(c)
        for cl_id in loaded:
            assert 'representatives' not in loaded[cl_id]

    def test_cluster_view_attribute_access(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=1, include_representatives=True,
        )
        cl = c[list(c)[0]]
        assert isinstance(cl.representatives, list)
        assert len(cl.representatives) > 0
        assert isinstance(cl.member_stats, dict)
        assert 'graph_outputs' in cl.member_stats


# ── to_cluster_infos ─────────────────────────────────────────────────

class TestToClusterInfos:

    def test_returns_list_of_dicts(self):
        c = make_synthetic_clustering(
            num_channels=2, clusters_per_channel=3, include_representatives=True,
        )
        infos = c.to_cluster_infos()
        assert isinstance(infos, list)
        assert len(infos) == 6

    def test_has_required_keys(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, include_representatives=True,
        )
        infos = c.to_cluster_infos()
        required_keys = {'index', 'channel_index', 'centroid', 'embeddings',
                         'index_tuples', 'graphs', 'image_paths'}
        for info in infos:
            assert required_keys.issubset(info.keys())

    def test_graphs_from_representatives(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=1,
            members_per_cluster=30, include_representatives=True,
        )
        infos = c.to_cluster_infos()
        info = infos[0]
        # Number of graphs equals number of representatives, not all members
        n_reps = len(c[list(c)[0]].representatives)
        assert len(info['graphs']) == n_reps
        # Each graph has the expected fields
        for g in info['graphs']:
            assert 'node_importances' in g
            assert 'edge_importances' in g
            assert 'graph_output' in g
            assert 'graph_deviation' in g
            assert 'graph_repr' in g
            assert 'node_indices' in g
            assert isinstance(g['node_importances'], np.ndarray)

    def test_without_representatives_gives_empty_graphs(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=1, include_representatives=False,
        )
        infos = c.to_cluster_infos()
        assert len(infos[0]['graphs']) == 0

    def test_embeddings_match_members(self):
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=1, members_per_cluster=40,
            include_representatives=True,
        )
        infos = c.to_cluster_infos()
        cl_id = list(c)[0]
        np.testing.assert_array_equal(infos[0]['embeddings'], c[cl_id].members)

    def test_round_trip_save_load_to_cluster_infos(self):
        """to_cluster_infos works identically on a loaded Clustering."""
        c = make_synthetic_clustering(
            num_channels=1, clusters_per_channel=2, include_representatives=True,
        )
        original_infos = c.to_cluster_infos()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'test.clu')
            c.save(path)
            loaded = Clustering.load(path)

        loaded_infos = loaded.to_cluster_infos()
        assert len(loaded_infos) == len(original_infos)
        for orig, load in zip(original_infos, loaded_infos):
            assert orig['index'] == load['index']
            assert orig['channel_index'] == load['channel_index']
            assert len(orig['graphs']) == len(load['graphs'])


# ── Report generation from Clustering ────────────────────────────────

class TestReportFromClustering:

    def test_report_generation_from_to_cluster_infos(self):
        """create_concept_cluster_report succeeds when fed to_cluster_infos() output.

        We use num_examples=0 to skip the image rendering (which needs actual
        molecule PNGs on disk). The histograms are still generated from the
        representative graphs' predictions, deviations, and importances.
        """
        from megan_global_explanations.visualization import create_concept_cluster_report

        c = make_synthetic_clustering(
            num_channels=2, clusters_per_channel=2,
            members_per_cluster=30, include_representatives=True,
        )
        infos = c.to_cluster_infos()

        with tempfile.TemporaryDirectory() as td:
            report_path = os.path.join(td, 'report.pdf')
            cache_path = os.path.join(td, 'cache')
            os.makedirs(cache_path)

            create_concept_cluster_report(
                cluster_data_list=infos,
                dataset_type='regression',
                path=report_path,
                cache_path=cache_path,
                num_examples=0,
                logger=logging.getLogger('test'),
            )

            assert os.path.exists(report_path)
            assert os.path.getsize(report_path) > 0

    def test_report_with_image_rendering(self):
        """Report renders example molecule images from to_cluster_infos() output.

        Creates minimal blank PNGs for draw_image and injects node_positions +
        edge_indices into the graph dicts so the importance-overlay rendering
        path is fully exercised.
        """
        from PIL import Image as PILImage
        from megan_global_explanations.visualization import create_concept_cluster_report

        num_channels = 1
        c = make_synthetic_clustering(
            num_channels=num_channels, clusters_per_channel=1,
            members_per_cluster=30, include_representatives=True,
        )
        infos = c.to_cluster_infos()

        with tempfile.TemporaryDirectory() as td:
            # Patch each representative graph with node_positions, edge_indices,
            # and a real image file on disk.
            for info in infos:
                for i, graph in enumerate(info['graphs']):
                    n_nodes = len(graph['node_indices'])
                    graph['node_positions'] = np.random.rand(n_nodes, 2).tolist()
                    # Build a simple chain of edges: 0-1, 1-2, ...
                    edge_indices = [[j, j + 1] for j in range(n_nodes - 1)]
                    graph['edge_indices'] = edge_indices
                    # Trim edge_importances to match the edge count
                    graph['edge_importances'] = np.random.rand(
                        len(edge_indices), num_channels
                    )

                    # Create a small blank PNG for draw_image
                    img_path = os.path.join(td, f'cl{info["index"]}_rep{i}.png')
                    PILImage.new('RGB', (64, 64), color='white').save(img_path)
                    info['image_paths'][i] = img_path

                # Also set index_tuples dataset_index to i so the file names are unique
                info['index_tuples'] = [(i, info['channel_index'])
                                        for i in range(len(info['graphs']))]

            report_path = os.path.join(td, 'report.pdf')
            cache_path = os.path.join(td, 'cache')
            os.makedirs(cache_path)

            create_concept_cluster_report(
                cluster_data_list=infos,
                dataset_type='regression',
                path=report_path,
                cache_path=cache_path,
                examples_type='random',
                num_examples=2,
                logger=logging.getLogger('test'),
            )

            assert os.path.exists(report_path)
            assert os.path.getsize(report_path) > 0
            # Check that example images were generated in the cache
            example_pngs = [f for f in os.listdir(cache_path) if 'example' in f]
            assert len(example_pngs) > 0
