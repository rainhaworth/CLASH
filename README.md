# CLASH
This repository contains the source code for the CLASH model, training scripts, and experiments submitted to RECOMB 2025.

## Example usage
We implement a data-independent hashing approach in `train-hash.py`, effectively creating a learned LSH function, and a data-dependent hashing approach in `train-hash-dependent.py`. Each can be used for all-vs-all and source-target search tasks. These examples use `db.fa` as the source dataset for both tasks and `ref.fa` as the target dataset for source-target tasks.

### Data-independent
```bash
python train-hash.py -l 1000 -s 500 -d 256 -m /path/to/model.h5
python hash.py -o src_index.pickle -l 1000 -d 256 -m /path/to/model.h5 -i db.fa
python hash.py -o tgt_index.pickle -l 1000 -d 256 -m /path/to/model.h5 -i ref.fa # source-target only
```
### Data-independent

#### All-vs-all
```bash
python train-hash-dependent.py -l 1000 -s 500 -d 128 -n 1 -m /path/to/model.h5 -t db.fa
python hash.py -o src_index.pickle -l 1000 -d 128 -n 1 -m /path/to/model.h5 -i db.fa
```
#### Source-target
```bash
python train-hash-dependent.py -l 1000 -s 500 -d 128 -n 1 -m /path/to/model.h5 -t ref.fa
python hash.py -o src_index.pickle -l 1000 -d 128 -n 1 -m /path/to/model.h5 -i db.fa
python hash.py -o tgt_index.pickle -l 1000 -d 128 -n 1 -m /path/to/model.h5 -i ref.fa
```

## Implementation
Our model architecture is is defined in `model/chunk_hash.py`. In total, our encoder uses a plain `Conv1D` layer, then 3 `InceptionLayer`s, and finally a single `ResBlock`, with `MaxPool1D(2)` between each layer. Filters are then `Flatten`ed and fed to a `Dense` hash layer with a `tanh` activation function. The objective function and custom hash metrics are implemented in `model/hash_metrics.py`. Synthetic data generators are implemented in `model/gensynth.py`. 

## Evaluation

### Ablation studies
Our evolution parameter ablation study is defined in `genassemble`, with `genassemble/evosim.py` defining evolution simulation, `genassemble/run.sh` performing the experiment, and `genassemble/figures.py` generating the figures shown in the paper.

Our training ablation across shared region thresholds and length-to-threshold ratios is performed by `train-hash-ablation.sh`. Evaluation at fixed shared region sizes can be performed by both training scripts by setting the `--band_eval` flag. If the model has already been trained, both scripts will skip to evaluation if the `--eval_only` flag is set.

Note that our scripts are designed for usage with SLURM on a specific HPC cluster and may require adaptation to function in other environments.

### BLAST validation
First, run BLAST on the source and target dataset, which may not be feasible for very large datasets. For all-vs-all, simply use the same dataset as source and target. We currently use the human-readable format and reduce the output with a script, which makes `eval-blast.py` implementation more convenient. We will likely change this in the future.
```bash
blastn -outfmt 0 -subject /path/to/source.fasta -query /path/to/target.fasta > blast-out.txt
utils/blast-reduce.py blast-out.txt > blast-reduced.txt
```

Finally, run `eval-blast.py`. This will determine whether each BLAST alignment has a matching hash collision, reporting the BLAST recall and search space remaining (SSR) used in the paper. We include all-vs-all and source-target examples below. The original model architecture and hashing approach are irrelevant beyond the chunk length `l` and the number of hash tables `n`.
```bash
python eval-blast.py --mode allvsall --srcindex src_index.pickle --blast blast-reduced.txt -n 8 -l 1000
python eval-blast.py --mode srctgt --srcindex src_index.pickle --tgtindex tgt_index.pickle --blast blast-reduced.txt -n 1 -l 1000
```
