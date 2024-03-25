=======================
Supplementary Materials
=======================

This folder contains the supplementary materials to go along with the paper. This file briefly describes the various 
additional materials.

==========================
Concept Clustering Reports
==========================

For each experiment, the generated concept explanations are automatically visualized as a ``concept_report.pdf`` file. In these 
PDF files, the information about each concept is aggregated and displayed in human-readable format across multiple pages. 
A header title declares the index of the concept as well as the corresponding explanation channel from which it was extracted
(for regression tasks, either "negative" or "positive"; for classification tasks, the class label). The subsequent pages 
show various information about the concept explanations, such as the number of members that have been identified as part of the 
cluster, the average contribution over all cluster members, the average mask size etc. Another page shows the visualizations of 
the cluster members. Each visualization is a combination of the original graph element and the explanation mask. The overall 
concept itself is defined as the underlying pattern that is shared among all the explanations that make up the concept cluster.
In addition to this, the concept prototype is given on another page, which is supposed to simplify the identification of this 
underlying pattern. The prototype graph, is the minimal graph structure, whose embedding still is reasonably close to the cluster 
centroid.

The concept cluster report PDF's are primarily sorted by the explanation channel from which they were extracted and in the 
second instance they are sorted by semantic similarity - meaning that two subsequent concepts listed in the PDF are the closests 
in regards to their centroid embedding.

- ``concept_report__ba2motifs.pdf``: The full, automatically generated concept report for the BA2Motifs dataset 
  that is discussed in the paper. This is a synthetically created graph classification dataset defined on randomly 
  generated BA graphs seeded with special subgraph motifs.
- ``concept_report__rb_dual_motifs.pdf``: The full, automatically generated concept report for the RbDualMotifs dataset 
  that is discussed in the paper. This is a synthetically created graph regression dataset defined on randomly generated 
  colored graphs seeded with special subgraph motifs.
- ``concept_report__mutagenicity.pdf``: The full, automatically generated concept report for the Mutagenicity dataset
  that is discussed in the paper. This is a real-world molecular classification dataset where molecular graphs have to
  be classified as either mutagenic or non-mutagenic to the DNA.
- ``concept_report__aqsoldb.pdf``: The full, automatically generated concept report for the AqSolDB dataset
  that is discussed in the paper. This is a real-world molecular regression dataset where molecular graphs have to
  be regressed to predict the solubility of the molecule in water.