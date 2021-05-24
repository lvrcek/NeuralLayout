# Neural Layout

Neural execution of algorithms used in Layout phase of the Overlap-Layout-Consensus genome assembly paradigm, based on MPNNs.

Overlap-Layout-Consensus is on the most common approaches to genome assembly with long reads obtained from the third-generation sequencers (mainly Oxford Nanopore and PacBio).
Here we focus on one part of the Layout phase, were an assembly graph is constructed from the overlapped sequences and path needs to be found through that graph.
At first, the graph is overly complicated and needs to be simplified. This is done by the initial simplification algorithms which detect and remove structures such as transitive edges, tips (dead ends), and bubbles (alternative paths between the same nodes), that are commonly foudn in the assembly graphs.

By relying on Pytorch Geometric, we construct an MPNN-based model to which would simulate the deterministic algorithms. This is a proof-of-concept work to show that graph neural networks can be used on assembly graphs.

## Requirements
#### Basic usage:
- Python >= 3.8
- Pip >= 19.2

#### Dependencies for Raven
- gcc 4.8+ | clang 4.0+
- cmake 3.11+
- zlib 1.2.8+

## Installation

First download the code:
```bash
git clone --recursive https://github.com/lvrcek/NeuralLayout.git
cd NeuralLayout
```

Create the virtual environment:
```bash
python -m venv env
source env/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

If you want to use Raven for creating the graphs from raw genomic reads in FASTQ format, run the following script.
If not, you can skip this part.
```bash
source setup.sh
```

## Usage

Basic dataset is already included in this repository. It consists of synthetic training data, synthetic testing data,
and real testing data obtained from the assembly graph of lambda phage. This enables you to run two
plug&play examples. For training the model on synthetic data and testing also on synthetic data, run:
```bash
python train.py --test_path data/test_synth
```
For training on synthetic data and testing on real lambda phage data, run:
```bash
python train.py --test_path data/test_real
```

You can also generate the training and testing data manually, by running:
```bash
python graph_generator.py data/train/raw --training
python graph_generator.py data/test/raw --testing
```

To test this model on some other reads in FASTQ format, put them into the `data/reads` directory.
For example, in case you `ecoli.fastq`, you should first run Raven assembler to generate graphs in CSV format,
and then create TXT files suitable for this model and save them into e.g. `data/test/raw`.
This can be done by running the following commands:
```bash
python graph_generator.py --from_fastq --fastq_path data/reads/ecoli.fastq --fastq_type ecoli data/csv
python graph_generator.py --from_csv --csv_path data/csv/ecoli.csv --csv_type ecoli data/test/raw
python train.py
```
This will first create the graph in the CSV format and save it into `data/csv` directory under the name `ecoli.csv`.
The second line will parse the `ecoli.csv` file and save it in a more appropriate format into the `data/test/raw` directory.
Finally, you run the training loop with default arguments `data/train` for train path and `data/test` for test path.


## Acknowledgement
This work was performed at Faculty of Electrical Engineering and Computing, University of Zagreb,
and Genome Institute of Singapore, A*STAR, as a part of the ARAP program. It was also partially
funded by the European Union through the European Regional Development Fund under the grant
KK.01.1.1.01.0009 (DATACROSS) and has been supported in part by the Croatian Science Foun-
dation under the project Single genome and metagenome assembly (IP-2018-01-5886) and “Young
Researchers” Career Development Program.