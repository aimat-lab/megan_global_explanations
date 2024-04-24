import os
import random
import tempfile
import logging
import typing as t

import umap
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
from megan_global_explanations.utils import DEFAULT_CHANNEL_INFOS


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


def concept_umap_visualization(concepts: list[dict],
                               graphs: list[dict],
                               channel_infos: dict[str, t.Any] = DEFAULT_CHANNEL_INFOS,
                               fidelity_threshold: t.Optional[float] = None,
                               num_neighbors: int = 100,
                               min_dist: float = 0.0,
                               spread: float = 1.0,
                               metric: str = 'manhattan',
                               repulsion_strength: float = 1.0,
                               random_state: int = 42,
                               plot_concepts: bool = True,
                               base_figsize: int = 5,
                               alpha: float = 0.3,
                               logger: logging.Logger = NULL_LOGGER,
                               ) -> tuple[plt.Figure, list[umap.UMAP]]:
    """
    Given a list of ``concepts`` and a list of ``graphs`` this function will create the 2D UMAP 
    visualizations of each explanation channels embeddings seperately. Optionally the locations of 
    concept cluster centroids will be added to these umapped visualizations as well.
    
    :param concepts: a list of dictionaries where each dictionary represents a concept cluster.
        These concept dictionaries have a pre-defined format.
    :param graphs: A list of graph dict representations. It is important that these graph dict 
        representations were already updated to include the optional "graph_embeddings" attribute
        as well. If the graphs do not include this property, the function will fail. This 
        attribute can be obtained by running a model forward pass.
    :param channel_infos: A dictionary that contains additional information about the channels
        which can be used for the visualization such as the name of the channel and the associated 
        color.
    :param fidelity_threshold: An optional threshold value that can be used to filter out graphs
        with a lower fidelity value than the given threshold. This filtering is only applied if the 
        this value is not None. In this case, the graph dict representations have to include the 
        optional "graph_fidelity" attribute as well.
    :param num_neighbors: The number of neighbors to use for the UMAP dimensionality reduction
    :param min_dist: The minimum distance to use for the UMAP dimensionality reduction
    :param spread: The spread to use for the UMAP dimensionality reduction
    :param metric: The metric to use for the UMAP dimensionality reduction
    :param repulsion_strength: The repulsion strength to use for the UMAP dimensionality reduction
    :param random_state: The random state to use for the UMAP dimensionality reduction
    :param plot_concepts: A boolean flag that indicates whether the concept centroids should be
        added to the visualization or not. If they are added, they will be visualized as crosess "x"
        scattered to the mapped location of the concept cluster centroids and also including a text 
        label with the cluster index.
    :param base_figsize: The base figure size to use for the visualization
    :param alpha: The alpha value to use for the scatter plot points
    :param logger: An optional logger instance to use for logging
    
    :returns: A tuple containing the matplotlib figure and a list of the umap mappers that were used
        for the dimensionality reduction
    """
    num_channels: int = len(set([concept['channel_index'] for concept in concepts]))
    
    fig, rows = plt.subplots(
        ncols=num_channels,
        nrows=1,
        figsize=(base_figsize * num_channels, base_figsize),
        squeeze=False,
    )
    
    # ~ UMAP dimensionality reduction
    # In this first step we need to create the dimensionality reduction with the umap algorithm for each 
    # of the channels separately. These mappings are based on the given list of embedding vectors.
    
    logger.info(f'creating UMAP visualizations for {num_channels} channels')
    mappers: list[umap.UMAP] = []
    for channel_index in range(num_channels):
        
        embeddings_channel = []
        for graph in graphs:
            
            if (fidelity_threshold is not None) and (graph['graph_fidelity'][channel_index] < fidelity_threshold):
                continue
            
            embeddings_channel.append(graph['graph_embeddings'][:, channel_index])
            
        # graph_embeddings_channel: (num_elements, num_dimensions)
        embeddings_channel = np.array(embeddings_channel)
        
        ax = rows[0][channel_index]
        # This dictionary contains additional information about the channel which can be used for 
        # the visualization such as the name of the channel and the associated color.
        channel_info: dict = channel_infos[channel_index]
        
        logger.info(f'* channel {channel_index}')
        
        # It is possible to pass in an external mapper instance by defining a custom 
        # attribute in the channel's information dict.
        if 'mapper' in channel_info:
            mapper = channel_info['mapper']
            
        # However, the default case is that we create a new mapper instance and fit it 
        # to the given embeddings.
        else:
            mapper = umap.UMAP(
                n_neighbors=num_neighbors,
                min_dist=min_dist,
                n_components=2,
                metric=metric,
                random_state=random_state,
                repulsion_strength=repulsion_strength,
                spread=spread,
            )
            logger.info('   fitting mapper...')
            mappings_channel = mapper.fit(embeddings_channel)
        
        mappers.append(mapper)
        mappings_channel = mapper.transform(embeddings_channel)

        logger.info('   plotting...')
        ax.scatter(
            mappings_channel[:, 0], mappings_channel[:, 1], 
            color=channel_info['color'],
            alpha=alpha,
            edgecolors='none',
        )
        ax.set_title(f'UMAP Projection\n'
                     f'Channel {channel_index} - {channel_info["name"]}')
        ax.set_xlabel('umap dimension 1')
        ax.set_ylabel('umap dimension 2')
        
    # ~ addding concepts
    # At this point we already have the umap projections for both of the channels, which are 
    # represented as 2D scatter plots with all the given embeddings. Now we want to add the 
    # concept information to those plots. For that we go through all the concepts and 
    # plot the projection of their centroids into the plots as well.
    
    if plot_concepts:
        
        logger.info('adding the concept centroids...')
        for concept in concepts:
            
            logger.info(f' * concept {concept["index"]}')
            # Based on the channel index with which the concept is associated we can then select 
            # the appropriate umap mapper and then project the centroid of the concept into the
            # corresponding figure
            channel_index = concept['channel_index']
            ax = rows[0][channel_index]
            mapper: umap.UMAP = mappers[channel_index]
            
            mapping = mapper.transform([concept['centroid']])[0]
            ax.scatter(
                mapping[0], mapping[1],
                color='black',
                marker='x',
            )
            ax.text(
                mapping[0], mapping[1],
                f'({concept["index"]})',
                color='black',
            )
    
    return fig, mappers


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
                                  normalize_centroid: bool = False,
                                  **kwargs,
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
            
            # 01.03.24 - backwards compatibility. Originally the concepts dictionaries were created in a way 
            # to have the two separate attributes "graphs" and "elements" which were lists containing the 
            # graph dict representations and the corresponding image paths of those graph elements. In the 
            # new version though, these are combined into the "elements" list which is a list of visual 
            # graph elements, that includes the image paths and the graphs!
            # So here we convert the one format into the other, if it exists.
            if 'elements' in data:
                data['graphs'] = [element['metadata']['graph'] for element in data['elements']]
                data['image_paths'] = [element['image_path'] for element in data['elements']]
            
            # ~ Basic cluster statistics
            num_elements = len(data['graphs'])
            channel_indices = [k for i, k in data['index_tuples']]
            # From all the individual channel indices we can create a soft assignment to a single channel
            channel_index = round(np.mean(channel_indices))
            
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
            ax.set_xlabel(f'Channel {channel_index} Fidelity')
            ax.set_ylabel(f'Number of Cluster Elements')
            ax.legend()
            contribution_path = os.path.join(temp_path, f'{cluster_index:02d}_contribution.png')
            fig.savefig(contribution_path, bbox_inches='tight')
            plt.close(fig)
            
            # The more important metric here is the size of the explanation mask
            mask_sizes = [np.sum(graph['node_importances']) for graph in graphs]
            mask_sizes_mean = np.mean(mask_sizes)
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(mask_sizes, bins=num_bins, color='lightgray')
            ax.axvline(mask_sizes_mean, color='black', label=f'avg: {mask_sizes_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.set_xlabel(f'Number of Nodes in Explanation Mask')
            ax.set_ylabel(f'Number of Cluster Elements')
            ax.legend()
            mask_size_path = os.path.join(temp_path, f'{cluster_index:02d}_mask_size.png')
            fig.savefig(mask_size_path, bbox_inches='tight')
            plt.close(fig)
            
            # ~ Prediction distribution
            # Another distribution that we want to look at is the distribution of the actual predicted values for all 
            # the cluster members. This will in general be less informative than the fidelity values but potentially 
            # still interesting for the contrast.
            
            # 01.03.24 - This was changed from just taking the prediction from the graph dict as it is. That would assume 
            # that the actual tensor that is output by the model was already properly processed according to the prediction 
            # type, which we should not assume and instead we should do that processing here: If the type is regression we 
            # just take the single value in the array and if classification we determine the class by argmax.
            predictions: t.List[float] = []
            for graph in graphs:
                pred = graph['graph_prediction']
                if isinstance(pred, np.ndarray):
                    if dataset_type == 'regression':
                        predictions.append(pred[0])
                    else:
                        predictions.append(np.argmax(pred))
                # backwards compatibility: If the prediction happens to be just a value then we assume this is already 
                # processed and we can just take it as it is.
                else:
                    predictions.append(pred)
            
            predictions_mean = np.mean(predictions)
            predictions_std = np.std(predictions)
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
            ax.hist(predictions, color='lightgray')
            ax.axvline(predictions_mean, color='black', label=f'avg: {predictions_mean:.2f}')
            x_min, x_max = ax.get_xlim()
            ax.set_xlim([min(0, x_min * lim_factor), max(0, x_max * lim_factor)])
            ax.set_xlabel(f'Model Prediction')
            ax.set_ylabel(f'Number of Cluster Elements')
            ax.legend()
            predictions_path = os.path.join(temp_path, f'{cluster_index:02d}_predictions.png')
            fig.savefig(predictions_path, bbox_inches='tight')
            plt.close(fig)
            
            # ~ Centroid and intra-cluster metrics
            # embeddings: (N, D) - just making sure that we are working with a numpy array here
            embeddings = np.array(data['embeddings'])
            # The centroid is just the mean of all those embeddings
            # centroid: (1, D)
            centroid = np.mean(embeddings, axis=0)
            # 31.01.24 
            # In some situations it makes sense to additionally normalize the centroid for example when the embeddings 
            # themselves are activated to be on the unit sphere then it would not make sense for the centroid not to be 
            # on it - yet a simple mean of all the embeddings would not guarantee the centroid also to be on the 
            # unit sphere.
            if normalize_centroid:
                centroid = centroid / np.linalg.norm(centroid)
            
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
            ax.set_xlabel(f'Distance to Centroid ({distance_func.__name__})')
            ax.set_ylabel(f'Number of Cluster Elements')
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
                    edge_importances=np.array(graph['edge_importances'])[:, k],
                )
                
                example_path = os.path.join(temp_path, f'{cluster_index:02d}_example_{i}.png')
                fig.savefig(example_path, bbox_inches='tight')
                plt.close(fig)
                
                examples.append({
                    'path': example_path,
                    'title': f'{i}'  
                })
                
            # ~ prototype visualizations
            # Optionally a cluster may also specify one or more "prototypes" these are subgraphs which are supposed 
            # to represent the underlying pattern of the concept cluster more clearly. 
            # These prototypes are passed in as visual graph elements in the "prototypes" list. Here we do a pre-processing 
            # were we create an image that also shows the models explanations for that prototype graph in addition to 
            # the raw graph image.
            
            if 'prototypes' in data:
                
                for i, _data in enumerate(data['prototypes']):
                    graph = _data['metadata']['graph']
                    image_path = _data['image_path']
                    
                    # There might be the chance that the prototype graphs do not contain explanation masks as part of their 
                    # attributes in which case we will skip the procedure to visualize those.
                    if ('node_importances' not in graph) or ('edge_importances' not in graph):
                        continue
                    
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
                        edge_importances=np.array(graph['edge_importances'])[:, k],
                    )
                    
                    prototype_path = os.path.join(temp_path, f'{cluster_index:02d}_prototype_{i}.png')
                    fig.savefig(prototype_path, bbox_inches='tight')
                    plt.close(fig)
                    
                    # Now we need to add that new path to the metadata of the prototype itself so that we can then 
                    # later access this path during the actual rendering of the report HTML.
                    _data['path'] = prototype_path
            
            # ~ Creating the html
            # After we have made all the previous computations we can create the HTML string that 
            # will visualize all that information.
            cluster_template = TEMPLATE_ENV.get_template('cluster_details.html.j2')
            cluster_string = cluster_template.render({
                **data,
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

    
    
