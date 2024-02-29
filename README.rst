|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/

=============
Global Concept-Explanations for the Self-Explaining MEGAN Graph Neural Network
=============

This package implements the functionality needed to extract global concept-based explanations from the recently published 
MEGAN (`GitHub <https://github.com/aimat-lab/graph_attention_student>`, `paper <https://link.springer.com/chapter/10.1007/978-3-031-44067-0_18>`) 
graph neural network model. The MEGAN model itself is a self-explaining graph neural network, which is able to
provide local attributional explanations its own predictions through an attention mechanism. By extending it's architecture and 
training process, it is possible to additionally extract concept-based explanations from it's latent space of explanation embeddings.
These concept explanations provide a *global* understanding of the model's decision making process.

**Abstract** Besides improving trust and validating model fairness, xAI
practices also have the potential to recover valuable scientific insights in
application domains where little to no prior human intuition exists. To
that end, we propose a method to create global concept explanations for
graph prediction tasks with the ultimate objective of gaining a deeper
understanding of graph predictive tasks, such as chemical property 
predictions. To achieve this we introduce the updated Megan2 version
of the recently introduced multi-explanation graph attention network.
Concept explanations are extracted by identifying dense clusters in the
model’s latent space of explanations. Finally, we optimize sub-graph 
prototypes to represent each concept cluster and optionally query a language
model to propose potential hypotheses for the underlying causal 
reasoning behind the identified structure-property relationships. We conduct
computational experiments on synthetic and real-world graph property
prediction tasks. For the synthetic tasks we find that our method correctly 
reproduces the structural rules by which they were created. For
real-world molecular property regression and classification tasks, we find
that our method rediscovers established rules of thumb as well as 
previously published hypotheses from chemistry literature. Additionally, the
concepts extracted by our method indicate more fine-grained resolution
of structural details than existing explainability methods. Overall, we
believe our positive results are a promising step toward the automated
extraction of scientific knowledge through AI models, suitable for more
complex downstream prediction tasks in the future

🔔 News
=======

- **March 2024** Paper is submitted to the `2nd xAI world conference <https://xaiworldconference.com/2024/>`

❓ What are Global Concept Explanations?
========================================

*Local* explanations aim to provide additional information about individual model predictions. Although there are different forms 
of local explanations the, the most common modality is that of importance attribution masks. For graph neural networks, these masks 
are defined on the node and edge level and usually provide a 0 to 1 *importance value* of how much a certain node or edge contributed
to the final prediction. While these explanations are very useful for understanding the model's decision making process on a case by 
case basis, it is hard to understand the model's general behavior.

*Global* explanations on the other hand aim to provide a more general understanding of the model's overal decision making process. As 
with local explanations, there exist different formats in which global model information can be presented, including generative explanations,
prototype-based explanations and concept-based explanations among others.

*Concept-based* explanations are one specific form of global explanations, which try to explain a models general behavior which is aggregated 
over many individual instances. The basic idea is to identify certain generalizable *concepts* which are then connected to a certain impact 
toward the model's prediction outcome. One such concept is generally defined as a common underlying pattern that is shared among multiple instances 
of the dataset. From a technical perspective, a concept can be defined as a set of input fragments. What exactly these input fragments are differs 
between application domains. In image processing, for example, these fragments are super pixels or image segments and in langauge processing they 
can be words or phrases. In the graph processing domain, these input fragments are subgraph motifs which can be contained in multiple different 
graphs of the dataset.

📦 Installation by Source
=========================

.. code-block:: shell

    git clone https://github.com/the16thpythonist/megan_global_explanations

Then in the main folder run a ``pip install``:

.. code-block:: shell

    cd megan_global_explanations
    python3 -m pip install -e .

Afterwards, you can check the install by invoking the CLI:

.. code-block:: shell

    python3 -m megan_global_explanations.cli --version
    python3 -m megan_global_explanations.cli --help


📌 Dependencies
===============

This package heavily builds on the following two packages:

- `visual_graph_datasets <https://github.com/aimat-lab/visual_graph_datasets/>`: This builds the epic story
- `graph_attention_student <https://github.com/aimat-lab/graph_attention_student/>`: 

🚀 Quickstart
=============



🧪 Computational Experiments
============================



📖 Referencing
==============

If you use, extend or otherwise mention or work, please cite the paper as follows:

.. code-block:: bibtex

    @article{teufel2024meganGlobal
        title={Global Concept-Explanations for the Self-Explaining MEGAN Graph Neural Network},
        author={Teufel, Jonas and Friederich, Pascal},
        journal={arxiv},
        year={2024}
    }

Credits
=======

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _MEGAN: https://link.springer.com/chapter/10.1007/978-3-031-44067-0_18 