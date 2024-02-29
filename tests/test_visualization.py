import os 
import io
import random
import tempfile
import shutil
import typing as t

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from weasyprint import HTML
from lorem_text import lorem

from visual_graph_datasets.visualization.base import create_frameless_figure, draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_background
from visual_graph_datasets.visualization.importances import plot_edge_importances_background
from megan_global_explanations.visualization import create_concept_cluster_report
from megan_global_explanations.visualization import generate_contrastive_colors

from .util import ARTIFACTS_PATH
from .util import load_mock_clusters
from .util import load_mock_vgd

    
def test_generate_contrastive_colors():
    """
    If the generation of colors with a high contrast works
    """
    fig, rows = plt.subplots(ncols=3, nrows=1, figsize=(30, 10), squeeze=False)
    
    for c, ax in enumerate(rows[0]):
        num = 3**(c+1)
        colors = generate_contrastive_colors(num)
        
        for i, color in enumerate(colors):
            x, y = random.uniform(0, 1), random.uniform(0, 1)
            ax.scatter(x, y, color=color)
            
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_generate_contrastive_colors.pdf')
    fig.savefig(fig_path)

    
def test_create_concept_cluster_report_basically_works():
    """
    The function create_concept_cluster should create a PDF file which visualizes the results of a concept 
    clustering operation in a visually pleasing way to the user.
    """
    num_clusters = 3
    output_path = os.path.join(ARTIFACTS_PATH, 'test_create_concept_cluster_report_baisically_works.pdf')
    
    cluster_data_list = load_mock_clusters()
    
    create_concept_cluster_report(
        cluster_data_list=cluster_data_list,
        path=output_path,
    )
    

def test_create_concept_cluster_report_centroid_examples_works():
    """
    With the option "centroid", the examples should be generated as those elements that are closest to 
    the centroid instead of just randomly chosen from the entire cluster.
    """
    num_clusters = 3
    output_path = os.path.join(ARTIFACTS_PATH, 'test_create_concept_cluster_report_centroid_examples_works.pdf')
    
    cluster_data_list = load_mock_clusters()
    
    create_concept_cluster_report(
        cluster_data_list=cluster_data_list,
        path=output_path,
        examples_type='centroid',
        num_examples=16,
    )


def test_create_concept_cluster_report_cache_path_works():
    """
    It should be possible to provide a "cache_path" option to the create_concep_cluster_report function 
    to explicitly determine where the temporary files during the creation process should be stored into
    """
    cluster_data_list = load_mock_clusters()
    
    with tempfile.TemporaryDirectory() as temp_path:
        
        cache_path = os.path.join(ARTIFACTS_PATH, 'test_create_concept_cluster_report_cache_path_works')
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        
        os.mkdir(cache_path)
        
        output_path = os.path.join(temp_path, 'report.pdf')
        create_concept_cluster_report(
            cluster_data_list=cluster_data_list,
            path=output_path,
            cache_path=cache_path,
        )
    
        assert os.path.exists(cache_path)
        assert os.path.isdir(cache_path)
        files = os.listdir(cache_path)
        assert len(files) != 0
        
        
def test_create_cluster_report_prototype_works():
    
    num_channels = 2
    num_clusters = 3
    index_data_map = load_mock_vgd()
    
    # This function will load a cluster data list from the disk where the dict elements exactly have the 
    # format that is required for the create_concept_cluster_report function
    cluster_data_list: t.List[dict] = load_mock_clusters(
        num_channels=num_channels,
        num_clusters=num_clusters,
    )
    
    # Now additionally to that base functionality, we want to test here the optional additional functionality 
    # of providing a cluster prototype as well.
    # A cluster prototype mainly has to define the path to the image which will be used as the actual prototype 
    # visualization and it may also provide a description string and a hypothesis string.
    # as the image path we are simply choose one of the images that are already used as an example.
    
    for cluster_info in cluster_data_list:
        # We just add a random graph from the cluster as the prototype
        prototype = random.choice(list(index_data_map.values()))
        prototype_graph = prototype['metadata']['graph']
        prototype_graph['node_importances'] = np.random.random(size=(len(prototype_graph['node_indices']), num_channels))
        prototype_graph['edge_importances'] = np.random.random(size=(len(prototype_graph['edge_indices']), num_channels))
        
        cluster_info['prototypes'] = [prototype]
    
    output_path = os.path.join(ARTIFACTS_PATH, 'test_create_concept_cluster_report_prototype_works.pdf')
    create_concept_cluster_report(
        cluster_data_list=cluster_data_list,
        path=output_path,
        examples_type='centroid',
        num_examples=16,
    )

    
    
def test_plot_importances_background():
    """
    If the plotting of node importances as a background highlight works properly.
    """
    # loading the sample element
    index_data_map = load_mock_vgd()
    
    data = index_data_map[2]
    graph = data['metadata']['graph']
    node_positions = graph['node_positions']
    ni, ei = graph['node_importances_2'], graph['edge_importances_2']
    
    # drawing the importances
    fig, ax = create_frameless_figure(width=1000, height=1000)
    draw_image(ax=ax, image_path=data['image_path'])
    plot_node_importances_background(
        ax=ax,
        g=graph,
        node_positions=node_positions,
        node_importances=ni[:, 1],
    )
    plot_edge_importances_background(
        ax=ax,
        g=graph,
        node_positions=node_positions,
        edge_importances=ei[:, 1],
    )
    
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_plot_importances_background.pdf')
    fig.savefig(fig_path)
    
    
def test_weasyprint():
    
    buffer = io.BytesIO()
    
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hello World</title>
    </head>
    <body>
        <h1>Hello World</h1>
        <p>Hello World!!</p>
        
        <p style="page-break-before: always"></p>
        
        <h1>Hello World</h1>
        <p>Hello World!!</p>
    </body>
    </html>
    """
    
    buffer.write(HTML(string=content).write_pdf())

    
    pdf_path = os.path.join(ARTIFACTS_PATH, 'test_weasyprint.pdf')
    with open(pdf_path, mode='wb') as file:
        file.write(buffer.getvalue())