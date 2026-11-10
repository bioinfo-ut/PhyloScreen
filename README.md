# PhyloScreen

A command-line tool for predicting bacterial phenotypes (e.g., antimicrobial resistance) from whole-genome sequences using [Mash](https://github.com/marbl/Mash) screen and logistic regression.

## Overview

PhyloScreen works in two stages:

1. **Train** — Build a Mash sketch from labeled reference genomes, extract features from pairwise Mash screen results, and train one logistic regression classifier per phenotype.
2. **Predict** — Screen a query genome once against the reference sketch and classify every phenotype in the model bundle.

A single model bundle can cover any number of phenotypes (for example one column per antibiotic), and reference isolates may have missing values for some of them.

### Features used for classification

Features are computed separately for each phenotype, using only the reference isolates that have a value for that phenotype:

- **Top hit phenotype** — phenotype label of the closest reference genome
- **Shared hashes ratio** — ratio of shared hashes between the top phenotype-1 hit and the top phenotype-0 hit
- **Weighted vote (top 10)** — shared-hash-weighted phenotype vote from the 10 closest references

## Requirements

- Python >= 3.13
- [Mash](https://github.com/marbl/Mash) >= 2.3

## Installation

```bash
git clone https://github.com/bioinfo-ut/PhyloScreen.git
cd PhyloScreen

# Create the Conda environment (installs Mash via bioconda)
conda env create -f environment.yml
conda activate phyloscreen

# Install the Python package
poetry install
```

## Usage

### Training a model

Prepare a tab-separated reference file (`refs.tsv`) with a `path` column followed by one or more phenotype columns. Every column other than `path` is treated as a phenotype, named by its header:

```
path	ampicillin	cefotaxime	ciprofloxacin
/data/genomes/sample_001.fa	1	0	
/data/genomes/sample_002.fa	0		1
/data/genomes/sample_003.fa	1	1	0
...
```

Each phenotype value is `1` (positive, e.g., resistant), `0` (negative, e.g., susceptible) or blank (unknown). An isolate with a blank value is still included in the Mash sketch, but for that phenotype it is excluded from training and is never chosen as a nearest neighbour; it is used normally for the other phenotypes.

Every phenotype needs both classes present, with at least as many labelled isolates per class as there are cross-validation folds (3); otherwise training stops with an error naming the phenotype.

Run training:

```bash
phyloscreen train -i refs.tsv -o model_dir/
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-i`, `--input` | *(required)* | TSV file with column `path` plus one or more phenotype columns |
| `-o`, `--output` | *(required)* | Output directory for the trained model |
| `-k`, `--kmer_size` | 21 | K-mer size for Mash sketch |
| `-s`, `--sketch_size` | 100000 | Sketch size for Mash sketch |
| `-t`, `--threads` | 4 | Number of threads |
| `--seed` | None | Random seed for reproducibility |
| `--verbose` | False | Enable verbose logging |

### Predicting phenotypes

```bash
phyloscreen predict -q query.fasta -m model_dir/ -o result.tsv
```

The query can be a single FASTA/FASTQ file or a directory containing multiple files (e.g. multiple read files for one isolate). Supported extensions: `.fasta`, `.fa`, `.fna`, `.fastq`, `.fq` (and their `.gz` variants).

To predict many isolates in one run, give a tab-separated list instead of a single query. The list needs a `path` column (one isolate per row; a row may point to a file or to a directory of read files for that isolate) and may have an optional `id` column. Blank or absent ids default to the file stem (or the directory name):

```
path	id
/data/queries/iso_001.fa	iso_001
/data/queries/iso_002_reads/	iso_002
/data/queries/iso_003.fq.gz	
```

```bash
phyloscreen predict -l queries.tsv -m model_dir/ -o results.tsv
```

The model is loaded once and Mash screen runs once per isolate. If an isolate cannot be processed (missing file, empty directory, Mash error), its rows are written with `prediction` set to `error`, the remaining isolates are still predicted, and the command exits with code 1 after listing the failed isolates.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--query` | *(one of -q / -l)* | Query genome file or directory (one isolate) |
| `-l`, `--query-list` | *(one of -q / -l)* | TSV listing many isolates (`path`, optional `id`) |
| `-m`, `--model` | *(required)* | Path to trained model directory |
| `-o`, `--output` | *(required)* | Output file path (TSV) |
| `-t`, `--threads` | 4 | Number of threads |
| `--verbose` | False | Enable verbose logging |

### Output format

The prediction output is a TSV file with one row per isolate and phenotype, with columns:

| Column | Description |
|--------|-------------|
| `query_id` | Isolate id (from the list file, otherwise the file stem or directory name) |
| `query_path` | Path to the query genome |
| `phenotype` | Phenotype name (column header from the training TSV) |
| `prediction` | Predicted phenotype (1 or 0; `unknown` if the query had no Mash hits; `error` if the isolate could not be processed) |
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
- `model.joblib` — Dictionary of trained scikit-learn pipelines (StandardScaler + LogisticRegression), one per phenotype
- `labels.tsv` — Reference genome paths and all phenotype columns (blank = missing)
- `coefficients.json` — Logistic regression feature coefficients per phenotype
- `metadata.json` — Phenotype names, feature names, Mash parameters and the selected regularisation strength per phenotype
- `screen_results/` — Per-reference Mash screen results used during training

Model directories created by PhyloScreen 1.0.0 (single `phenotype` column, no `metadata.json`) are still loaded by `predict` and reported as a phenotype named `phenotype`.
