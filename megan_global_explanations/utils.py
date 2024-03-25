import os
import shutil
import pathlib
import logging
import string
import tempfile
import random
import subprocess
from typing import List
import typing as t

import click
import jinja2 as j2
import numpy as np
from sklearn.metrics import pairwise_distances
from graph_attention_student.training import EpochCounterCallback


PATH = pathlib.Path(__file__).parent.absolute()
VERSION_PATH = os.path.join(PATH, 'VERSION')
EXPERIMENTS_PATH = os.path.join(PATH, 'experiments')
TEMPLATES_PATH = os.path.join(PATH, 'templates')

# Use this jinja2 environment to conveniently load the jinja templates which are defined as files within the
# "templates" folder of the package!
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_PATH),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'enumerate': enumerate,
})
TEMPLATE_ENV.filters.update(**{
    'nl2br': lambda value: value.replace('\n', '<br>'),
})

# This logger can be conveniently used as the default argument for any function which optionally accepts
# a logger. This logger will simply delete all the messages passed to it.
NULL_LOGGER = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


# == CLI RELATED ==

def get_version():
    """
    Returns the version of the software, as dictated by the "VERSION" file of the package.
    """
    with open(VERSION_PATH) as file:
        content = file.read()
        return content.replace(' ', '').replace('\n', '')


# https://click.palletsprojects.com/en/8.1.x/api/#click.ParamType
class CsvString(click.ParamType):

    name = 'csv_string'

    def convert(self, value, param, ctx) -> List[str]:
        if isinstance(value, list):
            return value

        else:
            return value.split(',')


# == STRING UTILITY ==
# These are some helper functions for some common string related problems

def safe_int(value: str) -> t.Optional[int]:
    """
    Given the string `value` this function will try to convert it to an integer. 
    If the string is not a valid integer, the function will return None.
    
    :param value: The string to be converted to an integer
    
    :returns: The integer value of the string or None if the string is not a valid integer
    """
    try:
        return int(value)
    except ValueError:
        return None


def random_string(length: int,
                  chars: string.ascii_letters + string.digits
                  ) -> str:
    """
    Generates a random string with ``length`` characters, which may consist of any upper and lower case
    latin characters and any digit.

    The random string will not contain any special characters and no whitespaces etc.

    :param length: How many characters the random string should have
    :param chars: A list of all characters which may be part of the random string
    :return:
    """
    return ''.join(random.choices(chars, k=length))


# == LATEX UTILITY ==
# These functions are meant to provide a starting point for custom latex rendering. That is rendering latex
# from python strings, which were (most likely) dynamically generated based on some kind of experiment data

def render_latex(kwargs: dict,
                 output_path: str,
                 template_name: str = 'article.tex.j2'
                 ) -> None:
    """
    Renders a latex template into a PDF file. The latex template to be rendered must be a valid jinja2
    template file within the "templates" folder of the package and is identified by the string file name
    `template_name`. The argument `kwargs` is a dictionary which will be passed to that template during the
    rendering process. The designated output path of the PDF is to be given as the string absolute path
    `output_path`.

    **Example**

    The default template for this function is "article.tex.j2" which defines all the necessary boilerplate
    for an article class document. It accepts only the "content" kwargs element which is a string that is
    used as the body of the latex document.

    .. code-block:: python

        import os
        output_path = os.path.join(os.getcwd(), "out.pdf")
        kwargs = {"content": "$\text{I am a math string! } \pi = 3.141$"
        render_latex(kwargs, output_path)

    :raises ChildProcessError: if there was ANY problem with the "pdflatex" command which is used in the
        background to actually render the latex

    :param kwargs:
    :param output_path:
    :param template_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as temp_path:
        # First of all we need to create the latex file on which we can then later invoke "pdflatex"
        template = TEMPLATE_ENV.get_template(template_name)
        latex_string = template.render(**kwargs)
        latex_file_path = os.path.join(temp_path, 'main.tex')
        with open(latex_file_path, mode='w') as file:
            file.write(latex_string)

        # Now we invoke the system "pdflatex" command
        command = (f'pdflatex  '
                   f'-interaction=nonstopmode '
                   f'-output-format=pdf '
                   f'-output-directory={temp_path} '
                   f'{latex_file_path} ')
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise ChildProcessError(f'pdflatex command failed! Maybe pdflatex is not properly installed on '
                                    f'the system? Error: {proc.stdout.decode()}')

        # Now finally we copy the pdf file - currently in the temp folder - to the final destination
        pdf_file_path = os.path.join(temp_path, 'main.pdf')
        shutil.copy(pdf_file_path, output_path)



def sort_cluster_centroids(cluster_centroid_map: dict,
                           metric: str = 'manhattan'):
    centroids = np.array(list(cluster_centroid_map.values()))
    distances = pairwise_distances(centroids, centroids, metric=metric)
    
    index_label_map = dict(enumerate(cluster_centroid_map.keys()))
    
    # This map will be the result of the function. The key of this dict is the original cluster label 
    # in which the clusters are labeled in the input cluster_centroid_map. THe value is the new label 
    # that is assigned to it through the centroid-based sorting
    label_map: t.Dict[int, int] = {0: 0}
    indices_inserted = set([0])
    current_index: int = 0
    for i, (label, centroid) in enumerate(cluster_centroid_map.items()):
        
        sorted_indices = np.argsort(distances[current_index])[1:]
        for j in sorted_indices:
            if j not in indices_inserted:
                label_mod = index_label_map[j]
                indices_inserted.add(j)
                label_map[label_mod] = i + 1
                current_index = j
                break

    return label_map


class RecordIntermediateEmbeddingsCallback(EpochCounterCallback):
    
    def __init__(self, 
                 epoch_step: int,
                 elements: t.List[t.Any],
                 active: bool = True,
                 logger: logging.Logger = NULL_LOGGER,
                 embedding_func: t.Callable = lambda mdl, els: mdl.embedd_graphs(els),
                 ):
        EpochCounterCallback.__init__(self)
        self.epoch_step = epoch_step
        self.elements = elements
        self.active = active
        self.logger = logger
        self.embedding_func = embedding_func
        
        self.epoch_embeddings_map = {}
        
    def on_epoch_end(self, epoch, logs=None):
        if self.active and self.epoch % self.epoch_step == 0:
            
            # embeddings: (B, D) or (B, D, K) depending whether its a multi channel model or not
            embeddings = self.embedding_func(self.model, self.elements)
            self.epoch_embeddings_map[self.epoch] = embeddings
            
            self.logger.info(f' * stored intermediate embeddings at epoch {self.epoch}')
            
            
# == GRAPH UTILITY ==

def extend_graph_info(index_data_map: dict):
    """
    This function extends the information contained in the graph dicts within the given ``index_data_map`` 
    representation of a visual graph dataset. In reality, this function will simply copy certain information
    from the metadata of the elements direclty into the graph dict, which is needed for some processing 
    steps.
    
    :param index_data_map: The visual graph dataset representation
    
    :returns: None, the modification is done in place.
    """
    for index, data in index_data_map.items():
        graph = data['metadata']['graph']
        graph['graph_index'] = data['metadata']['index']
        
        if 'repr' in data['metadata']:
            graph['graph_repr'] = data['metadata']['repr']
        else:
            graph['graph_repr'] = data['metadata']['value']