import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
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
    
    
def test_torch_checkpointing():
    """
    This test checks if the torch checkpointing works as expected. This is important because
    the checkpointing is used to save the model state and then load it again. Here we need to make sure 
    that is it possible to save a lightning model without having to use a Trainer instance as an intermediate!
    """
    class Net(pl.LightningModule):
        def __init__(self, flag: bool = False):
            super(Net, self).__init__()
            self.fc = nn.Linear(10, 1)
            self.flag = flag
            
            self.hparams.update({
                'flag': self.flag,
            })
            
        def forward(self, x):
            return self.fc(x)
    
    net = Net(flag=True)
    input_tensor = torch.rand(1, 10, requires_grad=False)
    output_tensor = net(input_tensor)
    
    # Save the model
    checkpoint_path = os.path.join(ARTIFACTS_PATH, 'model.ckpt')
    torch.save({
        'state_dict': net.state_dict(),
        'hyper_parameters': net.hparams,
        'pytorch-lightning_version': pl.__version__,
    }, checkpoint_path)
    
    # Load the model
    net_loaded = Net.load_from_checkpoint(checkpoint_path)
    assert net_loaded.flag == net.flag
    assert np.allclose(output_tensor.detach().numpy(), net_loaded(input_tensor).detach().numpy())