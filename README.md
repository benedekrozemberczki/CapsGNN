CapsGNN
============================================
A PyTorch implementation of "Capsule Graph Neural Network" (ICLR 2019).
<p align="center">
  <img width="800" src="CapsGNN.jpg">
</p>
<p align="justify">
The high-quality node embeddings learned from the Graph Neural Networks (GNNs) have been applied to a wide range of node-based applications and some of them have achieved state-of-the-art (SOTA) performance. However, when applying node embeddings learned from GNNs to generate graph embeddings, the scalar node representation may not suffice to preserve the node/graph properties efficiently, resulting in sub-optimal graph embeddings. Inspired by the Capsule Neural Network (CapsNet), we propose the Capsule Graph Neural Network (CapsGNN), which adopts the concept of capsules to address the weakness in existing GNN-based graph embeddings algorithms. By extracting node features in the form of capsules, routing mechanism can be utilized to capture important information at the graph level. As a result, our model generates multiple embeddings for each graph to capture graph properties from different aspects. The attention module incorporated in CapsGNN is used to tackle graphs with various sizes which also enables the model to focus on critical parts of the graphs. Our extensive evaluations with 10 graph-structured datasets demonstrate that CapsGNN has a powerful mechanism that operates to capture macroscopic properties of the whole graph by data-driven. It outperforms other SOTA techniques on several graph classification tasks, by virtue of the new instrument.</p>


This repository provides a PyTorch implementation of CapsGNN as described in the paper:

> Capsule Graph Neural Network.
> Zhang Xinyi, Lihui Chen.
> ICLR, 2019.
> [[Paper]](https://openreview.net/forum?id=Byl8BnRcYm)

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
torch-scatter     1.1.2
torch-sparse      0.2.2
torch-cluster     1.2.4
torch-geometric   1.0.3
torchvision       0.2.1
```
### Datasets
The code takes graphs for training from an input folder where each graph is stored as a JSON. Graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries are stored strings in order to make JSON serialization possible.

Every JSON file has the following key-value structure:

```javascript
{"edges": [[0, 1],[1, 2],[2, 3],[3, 4]],
 "labels": {"0": "A", "1": "B", "2": "C", "3": "A", "4": "B"},
 "target": 1}
```
The **edges** key has an edge list value which descibes the connectivity structure. The **labels** key has labels for each node which are stored as a dictionary -- within this nested dictionary labels are values, node identifiers are keys. The **target** key has an integer value which is the class membership.

### Outputs

The embeddings are saved in the `input/` directory. Each embedding has a header and a column with the node IDs. Finally, the node embedding is sorted by the node ID column.

### Options
Training a CapsGNN model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
```
#### Model options
```
  --seed               INT     Random seed.                       Default is 42.
  --number of walks    INT     Number of random walks per node.   Default is 10.
  --window-size        INT     Skip-gram window size.             Default is 5.
  --negative-samples   INT     Number of negative samples.        Default is 5.
  --walk-length        INT     Random walk length.                Default is 40.
  --lambd              FLOAT   Regularization parameter.          Default is 0.1
  --dimensions         INT     Number of embedding dimensions.    Default is 128.
  --workers            INT     Number of cores for pre-training.  Default is 4.   
  --learning-rate      FLOAT   SGD learning rate.                 Default is 0.025
```
### Examples
The following commands learn an embedding and save it with the persona map. Training a model on the default dataset.
```
python src/main.py
```
<p align="center">
  <img width="500" src="capsgnn.gif">
</p>

Training a Splitter model with 32 dimensions.
```
python src/main.py --dimensions 32
```
Increasing the number of walks and the walk length.
```
python src/main.py --number-of-walks 20 --walk-length 80
```
