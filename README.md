# PhenoScreen

A command-line tool for predicting bacterial phenotypes (e.g., antimicrobial resistance) from whole-genome sequences using [Mash](https://github.com/marbl/Mash) screen and logistic regression.

## Overview

PhenoScreen works in two stages:

1. **Train** — Build a Mash sketch from labeled reference genomes, extract features from pairwise Mash screen results, and train a logistic regression classifier.
2. **Predict** — Screen a query genome against the reference sketch and classify its phenotype using the trained model.

### Features used for classification

- **Top hit phenotype** — phenotype label of the closest reference genome
- **Shared hashes ratio** — ratio of shared hashes between the top phenotype-1 hit and the top phenotype-0 hit
- **Weighted vote (top 10)** — shared-hash-weighted phenotype vote from the 10 closest references

## Requirements

- Python >= 3.13
- [Mash](https://github.com/marbl/Mash) >= 2.3

## Installation

```bash
git clone https://github.com/bioinfo-ut/PhenoScreen.git
cd PhenoScreen

# Create the Conda environment (installs Mash via bioconda)
conda env create -f environment.yml
conda activate phenoscreen

# Install the Python package
poetry install
```

## Usage

### Training a model

Prepare a tab-separated reference file (`refs.tsv`) with columns `path` and `phenotype`:

```
path	phenotype
/data/genomes/sample_001.fa	0
/data/genomes/sample_002.fa	1
...
```

Where `phenotype` is `1` (positive, e.g., resistant) or `0` (negative, e.g., susceptible).

Run training:

```bash
phenoscreen train -i refs.tsv -o model_dir/
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | TSV file with columns: path, phenotype |
| `-o`, `--output` | *(required)* | Output directory for the trained model |
| `-k`, `--kmer_size` | 21 | K-mer size for Mash sketch |
| `-s`, `--sketch_size` | 100000 | Sketch size for Mash sketch |
| `-t`, `--threads` | 4 | Number of threads |
| `--seed` | None | Random seed for reproducibility |
| `--verbose` | False | Enable verbose logging |

### Predicting phenotypes

```bash
phenoscreen predict -q query.fasta -m model_dir/ -o result.tsv
```

The query can be a single FASTA/FASTQ file or a directory containing multiple files (e.g. multiple read files for one isolate). Supported extensions: `.fasta`, `.fa`, `.fna`, `.fastq`, `.fq` (and their `.gz` variants).

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--query` | *(required)* | Query genome file or directory |
| `-m`, `--model` | *(required)* | Path to trained model directory |
| `-o`, `--output` | *(required)* | Output file path (TSV) |
| `-t`, `--threads` | 4 | Number of threads |
| `--verbose` | False | Enable verbose logging |

### Output format

The prediction output is a TSV file with columns:

| Column | Description |
|--------|-------------|
| `query_path` | Path to the query genome |
| `prediction` | Predicted phenotype (1 or 0) |
| `probability` | Probability of phenotype 1 |
| `top_hit_1` | Closest reference with phenotype 1 |
| `top_hit_1_identity` | Mash identity to that reference |
| `top_hit_1_shared_hashes` | Shared hashes with that reference |
| `top_hit_0` | Closest reference with phenotype 0 |
| `top_hit_0_identity` | Mash identity to that reference |
| `top_hit_0_shared_hashes` | Shared hashes with that reference |

## Model artifacts

After training, the output directory contains:

- `reference_sketch.msh` — Mash sketch of all reference genomes
- `model.joblib` — Trained scikit-learn pipeline (StandardScaler + LogisticRegression)
- `labels.tsv` — Reference genome paths and their phenotype labels
- `coefficients.json` — Logistic regression feature coefficients
- `screen_results/` — Per-reference Mash screen results used during training
