import os
import random
import tempfile
import logging
import typing as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import euclidean
from weasyprint import HTML, CSS
from visual_graph_datasets.visualization.base import draw_image
from visual_graph_datasets.visualization.importances import plot_node_importances_border
from visual_graph_datasets.visualization.importances import plot_edge_importances_border
from visual_graph_datasets.visualization.importances import PLOT_NODE_IMPORTANCES_OPTIONS
from visual_graph_datasets.visualization.importances import PLOT_EDGE_IMPORTANCES_OPTIONS
from megan_global_explanations.utils import TEMPLATE_ENV, TEMPLATES_PATH
from megan_global_explanations.utils import NULL_LOGGER


def generate_contrastive_colors(num: int) -> t.List[str]:
    """
    Generate a list with a given ``num`` of matplotlib color tuples which have the highest contrast 
    
    :returns: a list of lists where each list contains the RGB float values for a color
    """
    hues = np.linspace(0, 0.9, num)
    colors = mcolors.hsv_to_rgb([[h, 0.7, 0.9] for h in hues])
    return colors


def plot_distributions(ax: plt.Axes,
                       xs: t.List[float],
                       values_list: t.List[t.List[float]],
                       color: str = 'black',
                       margin: int = 25,
                       line_style: str = '--',
                       fill: bool = True,
                       fill_alpha: float = 0.2,
                       ):
    
    medians = [np.median(values) for values in values_list]
    lowers = [np.percentile(values, margin) for values in values_list]
    uppers = [np.percentile(values, 100 - margin) for values in values_list]
    
    ax.plot(xs, medians, color=color)
    ax.plot(xs, lowers, color=color, ls=line_style)
    ax.plot(xs, uppers, color=color, ls=line_style)

    if fill:
        ax.fill_between(xs, lowers, uppers, color=color, alpha=fill_alpha, linewidth=0.0)


def animate_deepect_history(history: t.Dict[int, dict],
                            output_path: str,
                            fig_size: tuple = (10, 10),
                            fps: int = 5,
                            ):
    
    num_frames = len(history)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=fig_size)
    
    xs = np.array([emb[0] for clusters in history.values() for data in clusters for emb in data['embeddings']])
    ys = np.array([emb[1] for clusters in history.values() for data in clusters for emb in data['embeddings']])
    
    x_min, x_max = np.max(xs), np.min(xs)
    y_min, y_max = np.max(ys), np.min(ys)
    
    def update(frame):
        
        epoch, clusters = list(history.items())[frame]
            
        ax.clear()
        ax.set_title(f'Epoch: {epoch} - Frame: {frame}')
        
        for data in clusters:
            embeddings = np.array(data['embeddings'])
            if len(embeddings.shape) != 2:
                continue
            
            ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1],
                label=f'node {data["index"]} ({data["weight"]:.2f}): {len(data["embeddings"])}'
            )
            ax.scatter(
                data['center'][0], data['center'][1],
                color='black',
                marker='s',
                facecolor='none',
                s=100,
            )
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.legend()
    
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=num_frames,
        blit=False,
    )
    anim.save(output_path, writer='ffmpeg', fps=fps)
    return anim


def create_concept_cluster_report(cluster_data_list: t.List[dict],
                                  path: str,
                                  dataset_type: str = 'regression',
                                  examples_type: str = 'random',
                                  num_examples: int = 8,
                                  num_bins: int = 50,
                                  fig_size: tuple = (5, 5),
                                  logger: logging.Logger = NULL_LOGGER,
                                  plot_node_importances: t.Union[str, t.Callable] = 'background',
                                  plot_edge_importances: t.Union[str, t.Callable] = 'background',
                                  cache_path: t.Optional[str] = None,
                                  distance_func: t.Callable = euclidean,
                                  ) -> None:
    """
    Each cluster details dict has to consist of the following items:
    - index: An integer index that uniquely identifies the cluster
    - embeddings: a list of the D-dimensional cluster embeddings
    - index_tuples: 
    - graphs: A list of the graphs
    
    Note that all the given lists needs to be in the same order, which means that list elements 
    at the same indices need to represent information about the same dataset element.
    """
    report_template = TEMPLATE_ENV.get_template('cluster_report.html.j2')
    
    with tempfile.TemporaryDirectory() as temp_path:
        
        # "cache_path" is an optional parameter that can be used to explicitly provide a folder path 
        # into which all the temporary files should be saved into. This can be useful when the 
        # individual image files of the examples will be needed separately to the report PDF 
        # for example.
        if cache_path is not None and os.path.exists(cache_path):
            temp_path = cache_path
        
        num_clusters = len(cluster_data_list)
        logger.info(f'starting to create report for {num_clusters} clusters...')
        
        pages = []
        for p, data in enumerate(cluster_data_list):
            cluster_index = data['index']
            
            # ~ Basic cluster statistics
            num_elements = len(data['graphs'])
            channel_indices = [k for i, k in data['index_tuples']]
            
            # in the first step we want to create aggregation statistics for all the members of the cluster 
            # which most importantly includes the mask size and contribution to the prediction outcome
            graphs = data['graphs']
            image_paths = data['image_paths']
            index_tuples = data['index_tuples']
            
            # These deviatons are still 2-dim tensors for each of the graphs becasue they contain the output 
            # deviations for every pairing of output value and explanation channel.
            # for the visualization however we need to reduce this to a single value and which value to 
            # to choose in this case is different for regression and classification results
            # deviations: (B, K, C)
            deviations = [graph['graph_deviation'] for graph in graphs]
            contributions = []
            if dataset_type == 'regression':
                # Since regression is only one value anyways we only need to choose the appropriate channel 
                for graph, dev, (i, k) in zip(graphs, deviations, index_tuples):
                    contributions.append(dev[0][k])
                    
            if dataset_type == 'classification':
                for graph, dev, (i, k) in zip(graphs, deviations, index_tuples):
                    contributions.append(dev[k][k])
                    
            lim_factor = 1.1
            
            contributions_mean = np.mean(contributions)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(contributions, bins=num_bins, color='lightgray')
            ax.axvline(contributions_mean, color='black', label=f'avg: {contributions_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.legend()
            contribution_path = os.path.join(temp_path, f'{cluster_index:02d}_contribution.png')
            fig.savefig(contribution_path, bbox_inches='tight')
            
            # The more important metric here is the size of the explanation mask
            mask_sizes = [np.sum(graph['node_importances']) for graph in graphs]
            mask_sizes_mean = np.mean(mask_sizes)
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(mask_sizes, bins=num_bins, color='lightgray')
            ax.axvline(mask_sizes_mean, color='black', label=f'avg: {mask_sizes_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.legend()
            mask_size_path = os.path.join(temp_path, f'{cluster_index:02d}_mask_size.png')
            fig.savefig(mask_size_path, bbox_inches='tight')
            
            # ~ Prediction distribution
            # Another distribution that we want to look at is the distribution of the actual predicted values for all 
            # the cluster members. This will in general be less informative than the fidelity values but potentially 
            # still interesting for the contrast.
            predictions = [graph['graph_prediction'] for graph in graphs]
            predictions_mean = np.mean(predictions)
            predictions_std = np.std(predictions)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(predictions, color='lightgray')
            ax.axvline(predictions_mean, color='black', label=f'avg: {predictions_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.legend()
            predictions_path = os.path.join(temp_path, f'{cluster_index:02d}_predictions.png')
            fig.savefig(predictions_path, bbox_inches='tight')
            
            # ~ Centroid and intra-cluster metrics
            # embeddings: (N, D) - just making sure that we are working with a numpy array here
            embeddings = np.array(data['embeddings'])
            # The centroid is just the mean of all those embeddings
            # centroid: (1, D)
            centroid = np.mean(embeddings, axis=0)
            # Now with the centroid we can calculate the distances of all the embeddings and then create a plot that 
            # shows the distribution of those. this will be interesting to judge the homogenity of the cluster.
            centroid_distances = np.array([distance_func(centroid, emb) for emb in embeddings])
            centroid_distances_mean = np.mean(centroid_distances)
            centroid_distances_std = np.std(centroid_distances)
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(centroid_distances, color='lightgray')
            ax.axvline(centroid_distances_mean, color='black', label=f'avg: {centroid_distances_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.legend()
            centroid_distances_path = os.path.join(temp_path, f'{cluster_index:02d}_centroid_distances.png')
            fig.savefig(centroid_distances_path, bbox_inches='tight')
            plt.close(fig)
            
            # ~ Creating the example visualizations
            
            # The easiest default case is to just pick some random elements from the entire cluster. This will 
            # perhaps provide the most realistic overview over the cluster.
            indices = list(range(num_elements))
            num_examples_ = min(num_examples, len(indices))
            if examples_type == 'random':
                example_indices = random.sample(indices, k=num_examples_)
                
            # In this case we want to select the examples as those elements which are the clostest to the cluster 
            # centroid. For that purpose we obtain the index sorting of the distances to the centroid and then 
            # pick those elements with the lowest distances. 
            elif examples_type == 'centroid':
                sorted_indices = np.argsort(centroid_distances).tolist()
                example_indices = sorted_indices[:num_examples_]
            
            examples: t.List[dict] = []
            for c in example_indices:
                graph, image_path, (i, k) = graphs[c], image_paths[c], index_tuples[c]

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                draw_image(ax, image_path)
                
                plot_node_importances_func = PLOT_NODE_IMPORTANCES_OPTIONS[plot_node_importances]
                plot_node_importances_func(
                    ax=ax,
                    g=graph,
                    node_positions=np.array(graph['node_positions']),
                    node_importances=np.array(graph['node_importances'])[:, k]
                )
                
                plot_edge_importances_func = PLOT_EDGE_IMPORTANCES_OPTIONS[plot_edge_importances]
                plot_edge_importances_func(
                    ax=ax,
                    g=graph,
                    node_positions=np.array(graph['node_positions']),
                    edge_importances=np.array(graph['edge_importances'])[:, k]
                )
                
                example_path = os.path.join(temp_path, f'{cluster_index:02d}_example_{i}.png')
                fig.savefig(example_path, bbox_inches='tight')
                plt.close(fig)
                
                examples.append({
                    'path': example_path,
                    'title': f'{i}'  
                })
            
            # ~ Creating the html
            # After we have made all the previous computations we can create the HTML string that 
            # will visualize all that information.
            cluster_template = TEMPLATE_ENV.get_template('cluster_details.html.j2')
            cluster_string = cluster_template.render({
                'index': cluster_index,
                'num_elements': num_elements,
                'channel_index': {
                    'avg': np.mean(channel_indices),
                    'std': np.std(channel_indices),
                    'active': round(np.mean(channel_indices)),
                },
                'contribution': {
                    'path': contribution_path,
                    'avg': np.mean(contributions),
                    'std': np.std(contributions)
                },
                'mask_size': {
                    'path': mask_size_path,
                    'avg': np.mean(mask_sizes),
                    'std': np.std(mask_sizes),
                },
                'centroid_distance': {
                    'path': centroid_distances_path,
                    'avg': centroid_distances_mean,
                    'std': centroid_distances_std,
                },
                'prediction': {
                    'path': predictions_path,
                    'avg': predictions_mean,
                    'std': predictions_std,   
                },
                'values': {
                    'silhouette': 0,
                },
                'examples': examples,
            })
            
            pages.append(cluster_string)
            logger.info(f' * ({p+1}/{num_clusters}) done')
            
        report_string = report_template.render({
            'pages': pages
        })
        html = HTML(string=report_string)
        
        logger.info('converting to PDF...')
        cluster_css = CSS(os.path.join(TEMPLATES_PATH, 'cluster_details.css'))
        html.write_pdf(path, stylesheets=[cluster_css])

    
    