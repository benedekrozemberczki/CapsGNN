CapsGNN
============================================
A PyTorch implementation of "Capsule Graph Neural Network" (ICLR 2019).
<p align="center">
  <img width="500" src="capsgnn.png">
</p>
<p align="justify">
The high-quality node embeddings learned from the Graph Neural Networks (GNNs) have been applied to a wide range of node-based applications and some of them have achieved state-of-the-art (SOTA) performance. However, when applying node embeddings learned from GNNs to generate graph embeddings, the scalar node representation may not suffice to preserve the node/graph properties efficiently, resulting in sub-optimal graph embeddings. Inspired by the Capsule Neural Network (CapsNet), we propose the Capsule Graph Neural Network (CapsGNN), which adopts the concept of capsules to address the weakness in existing GNN-based graph embeddings algorithms. By extracting node features in the form of capsules, routing mechanism can be utilized to capture important information at the graph level. As a result, our model generates multiple embeddings for each graph to capture graph properties from different aspects. The attention module incorporated in CapsGNN is used to tackle graphs with various sizes which also enables the model to focus on critical parts of the graphs. Our extensive evaluations with 10 graph-structured datasets demonstrate that CapsGNN has a powerful mechanism that operates to capture macroscopic properties of the whole graph by data-driven. It outperforms other SOTA techniques on several graph classification tasks, by virtue of the new instrument. </p>


This repository provides a PyTorch implementation of CapsGNN as described in the paper:

> Capsule Graph Neural Network.
> Zhang Xinyi, Lihui Chen.
> ICLR, 2018.
> [[Paper]](https://arxiv.org/abs/1802.02896)

### Requirements
