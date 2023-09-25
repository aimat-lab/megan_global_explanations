import os

from megan_global_explanations.utils import get_version
from megan_global_explanations.utils import render_latex
from megan_global_explanations.utils import sort_cluster_centroids

from .util import ASSETS_PATH, ARTIFACTS_PATH


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)


def test_sort_cluster_centroids():
    
    # This is an example of a cluster labeling where the key is the cluster label and the value 
    # is the centroid vector for the cluster. The order of the labels is not pretty random
    cluster_centroid_map = {
        0: [10, 20],
        1: [-2, 3],
        2: [12, 19],
        3: [0, 1],
        4: [-3, 3],
        5: [9, 21],
        6: [5, 6],
    }
    
    # Now we want to re assign clusters such that there is some kind of order to them. 
    # Specifically we want to reduce the distance of the centroids of two cluster labels that 
    # immediately follow each other.
    # This is the goal that is achieved by the following function. It returns a dictionary which maps 
    # the original int cluster index to a new cluster index.
    label_mapping: dict = sort_cluster_centroids(cluster_centroid_map)
    
    cluster_centroid_map_sorted = {j: cluster_centroid_map[i] for i, j in label_mapping.items()}
    print('original', cluster_centroid_map)
    print('sorted', cluster_centroid_map_sorted)