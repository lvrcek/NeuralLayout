# Neural Layout

Neural execution of algorithms used in Layout phase of the Overlap-Layout-Consensus genome assembly paradigm, based on MPNNs.

Overlap-Layout-Consensus is on the most common approaches to genome assembly with long reads obtained from the third-generation sequencers (mainly Oxford Nanopore and PacBio).
Here we focus on one part of the Layout phase, were an assembly graph is constructed from the overlapped sequences and path needs to be found through that graph.
At first, the graph is overly complicated and needs to be simplified. This is done by the initial simplification algorithms which detect and remove structures such as transitive edges, tips (dead ends), and bubbles (alternative paths between the same nodes), that are commonly foudn in the assembly graphs.

By relying on Pytorch Geometric, we construct an MPNN-based model to which would simulate the deterministic algorithms. This is a proof-of-concept work to show that graph neural networks can be used on assembly graphs.


## Installation and running

First download the code:
```
git clone --recursive https://github.com/lvrcek/NeuralLayout.git
```

TODO: include the requirements file for pip installation

Generate the training and testing data by running:
```
python graph_generator.py data/train/raw --training
python graph_generator.py data/test/raw --testing
```

Run the training process:
```
python train.py
```

## Acknowledgement
This work was performed at Faculty of Electrical Engineering and Computing, University of Zagreb,
and Genome Institute of Singapore, A*STAR, as a part of the ARAP program. It was also partially
funded by the European Union through the European Regional Development Fund under the grant
KK.01.1.1.01.0009 (DATACROSS) and has been supported in part by the Croatian Science Foun-
dation under the project Single genome and metagenome assembly (IP-2018-01-5886) and “Young
Researchers” Career Development Program.