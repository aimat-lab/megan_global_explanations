# Changelog

## 0.4.0 - 2026-04-17

### Added

- **`Clustering` class** (`megan_global_explanations/data.py`): A self-contained data structure that
  holds all clustering state (per-channel embeddings, HDBSCAN labels, centroids, member embeddings,
  agglomerative linkage matrices, and extensible annotations) with a rich API:
  - `Clustering.load(path)` / `.save(path, max_members=2000)` — serializes to a `.clu` zip archive
    containing human-readable JSON metadata alongside binary `.npy`/`.npz` arrays
  - `.score(embedding, channel, method='knn', k=5)` — score an embedding against all clusters in a
    channel using k-NN distance or centroid distance
  - `.at_linkage(percent)` — return a lazy merged view at a given dendrogram cut height, with
    min-aggregation scoring across constituent leaf clusters
  - `clustering[cluster_id]` — dict-like + attribute access to individual clusters via `ClusterView`
  - `.get_tree(channel)` / `.get_forest()` — networkx DiGraph representation of the cluster hierarchy
  - `Clustering.from_experiment(...)` — construct from in-memory experiment data
- **`ClusterView`** (`megan_global_explanations/data.py`): lightweight dict-like + attribute-access
  wrapper returned by `Clustering[cluster_id]`
- **Post-hoc agglomerative clustering of clusters** (`vgd_concept_extraction.py`): new analysis stage
  that runs hierarchical clustering over existing leaf clusters per channel, using full member point
  clouds with average linkage under `CLUSTERING_METRIC`. Produces per-channel dendrogram PNGs,
  a pickled linkage matrix, and a textual merge summary at 10% increments (10%–90%) in the
  experiment log
- **Merged-cluster SMARTS overlap plot** (`vgd_concept_extraction__synth2.py`): when
  `CLUSTER_HIERARCHY_CUT` is set, produces a second SMARTS overlap heatmap
  (`smarts_cluster_overlap_merged.png`) using the super-clusters from the dendrogram cut, with
  labels like `ch0_[cl0+cl3+cl4+cl5]`
- **k-NN cluster scoring mode** (`vgd_concept_extraction__synth2.py`): new `CLUSTER_SCORE_METHOD='knn'`
  option that scores embeddings by the mean distance to the k nearest training-set cluster members.
  Shape-aware, smooth falloff, and unaffected by the `membership_vector` / leaf-selection degeneracy
- **Configurable scoring in `predict_and_score.py`**: the standalone example script now uses the
  `Clustering` class to load a `.clu` archive and score molecules via `.score()`. Supports `knn` and
  `distance` methods
- **Dynamic processing module loading** (`predict_and_score.py`): `PROCESSING_PATH` config points to a
  `process.py` module that is loaded at runtime via `importlib`, ensuring feature-encoding parity with
  the training dataset
- **Relative + absolute path support** (`predict_and_score.py`): all configured paths are resolved
  relative to the script's directory if not absolute
- **Representative members per cluster**: each cluster now stores N representative members
  (default 16) with their full explanation data (SMILES, node/edge importances, predictions,
  deviations) inside the `.clu` archive. This enables report generation from the archive alone,
  without needing the model or dataset at render time
- **`select_representatives()` function** (`data.py`): standalone utility for selecting
  representative members with two strategies:
  - `'closest'`: deterministic — the N nearest members to the centroid
  - `'temperature'` (default): stochastic softmax sampling over negative centroid-distances with
    auto-scaled temperature (multiplier of median intra-cluster distance), providing diversity
    while biasing toward typical members
- **Per-cluster member statistics** (`member_stats`): pre-computed arrays for histogram generation
  (graph outputs, deviations, mask sizes, centroid distances) stored in the `.clu` archive,
  enabling report histograms without re-running the model
- **`Clustering.to_cluster_infos()`**: bridge method that reconstructs the `cluster_infos`-style
  list of dicts expected by `create_concept_cluster_report`, using stored representatives as
  example graphs. Enables standalone report generation from a loaded `.clu` archive
- **Comprehensive test suite** (`tests/test_clustering.py`): 57 tests covering `ClusterView`,
  `Clustering` basics, save/load round-trips, scoring, `at_linkage()`, `get_tree()`/`get_forest()`,
  `select_representatives`, representatives save/load, `to_cluster_infos()`, and end-to-end report
  generation including image rendering

### New parameters

- `CLUSTER_SCORE_METHOD` (`'membership'` | `'distance'` | `'knn'`): selects the scoring function
  used by the `compute_cluster_score` hook throughout the experiment pipeline
- `CLUSTER_SCORE_KNN_K` (default 5): number of nearest members to average over in knn mode
- `ANALYZE_CLUSTER_HIERARCHY` (default True): toggle the post-hoc agglomerative clustering analysis
- `CLUSTER_HIERARCHY_LINKAGE` (default `'average'`): linkage method for the hierarchy
- `CLUSTER_HIERARCHY_CUT` (default 0.8): dendrogram cut fraction for the merged SMARTS overlap plot
- `NUM_REPRESENTATIVES` (default 16): number of representative members stored per cluster
- `REPRESENTATIVE_STRATEGY` (default `'temperature'`): how to select representatives
  (`'closest'` or `'temperature'`)
- `REPRESENTATIVE_TEMPERATURE` (default 0.7): temperature multiplier for the stochastic
  selection strategy (fraction of median intra-cluster distance)

### Changed

- The experiment now exports a `clustering.clu` archive alongside the existing `clusterers.pkl` and
  `centroids.json` artefacts. The `.clu` format is the recommended way to consume clustering results
  going forward
- `predict_and_score.py` now imports from `megan_global_explanations.data.Clustering` and no longer
  depends on HDBSCAN's `membership_vector` or direct access to `clusterer.prediction_data_`
- Added `networkx>=2.6` as a project dependency (used by `Clustering.get_tree()` /
  `.get_forest()`)

## 0.3.0 - 2025-12-04

- Added the `concept_extraction.py` experiment which is a base experiment that does essentially the same
  as the existing `vgd_concept_extraction.py` only that it does not need a pre-compiled visual graph dataset
- Added `concept_extraction__aqsdolb.py`
- The concept extraction experiments now explicitly export the concept centroids in a JSON file.
- Refactored the `README.rst` file to now list the experiments based on the simple CSV files
  first and only list the VGD based experiments as the second option

## 0.2.2 - 2024-09-16

- Fixed the Reader class to actually use the elements that were read with VisualGraphDatasetReader in the case of
  explicitly passing the concepts.

## 0.2.1 - 2024-09-13

- Modified the Reader and Writer classes to support the direct export and imports of the elements as visual graph elements
  in the format of a visual graph dataset folder.

## 0.2.0 - 2024-03-25

- Added the function `main.extract_concepts` function which performs the concept extraction / clustering for a given
  model and dataset combination and returns a list of all the identified concept dicts.
- Added the function `generate_concept_prototypes` which takes an existing list of concepts, the original model and the
  dataset as parameters and will apply a genetic algorithm optimization to generate prototype graphs.

## 0.1.0 - 2024-03-06

Initial version
