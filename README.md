<div align="center" style="margin: 0; padding: 0;">
  <h1 style="margin: 0; padding: 0;">🐍 scWGBS-GPT</h1>
  <img src="pic/scWGBS-GPT.png" width="240" style="display: block; margin: 0 auto;"/>
  <p>A Foundation Model for Capturing Long-Range CpG Dependencies in Single-Cell Whole-Genome Bisulfite Sequencing to Enhance Epigenetic Analysis</p>
   <div align="center" style="margin: 1.5rem 0;">
   <div style="display: flex; gap: 1.25rem; justify-content: center;">
      <a href="#-key-features" style="text-decoration: none; color: #2d3748; font-weight: 500;">Key Features</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-performance" style="text-decoration: none; color: #2d3748; font-weight: 500;">Performance</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-citation" style="text-decoration: none; color: #2d3748; font-weight: 500;">Citation</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-quick-start" style="text-decoration: none; color: #2d3748; font-weight: 500;">Quick Start</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-faq" style="text-decoration: none; color: #2d3748; font-weight: 500;">FAQ</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-license" style="text-decoration: none; color: #2d3748; font-weight: 500;">License</a>
      <span style="color: #cbd5e0;">•</span>
      <a href="#-contributing" style="text-decoration: none; color: #2d3748; font-weight: 500;">Contributing</a>
   </div>
   </div>
</div>


## 🌟 Highlights
🧬 **Analyzing Single-Cell Methylation Data** | 🌐 **Whole-Genome-Scale Context Modeling** | 🔬 **Single-CpG Resolution** | ⚡ **Mamba-Powered Speed**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org)

---

## 🔥 Key Features

- **🧬 Analyzing Single-Cell Methylation Data**: The first language model for analyzing single-cell methylation data, offering unparalleled accuracy and scalability.
- **🌐 Whole-Genome-Scale Processing**: Handles sequences with **up to 2 million CpG sites** - 100× longer than conventional methods
- **🔬 Single-CpG Resolution**: Captures methylation patterns at individual CpG level with 93.4% average cell type classification accuracy
- **⚡ Mamba-Powered Efficiency**: Combines selective state space models with cross-attention for **200× faster inference** vs standard transformers
- **🧩 Modular Design**: Easily adaptable for diverse epigenomic analysis tasks including:
  - Cell type annotation
  - Cancer subtyping
  - Developmental trajectory inference
  - Deconvolution cell-free DNA methylation data

## 🏆 Performance
**Human Brain Cell Type Classification**  
Test accuracy on 15 cell types from human prefrontal cortex:

| Cell Type  | Accuracy | Cell Type | Accuracy |
|------------|----------|-----------|----------|
| L2/3-IT    | 98.16%   | L6-CT     | 93.41%   |
| ODC        | 99.63%   | Foxp2     | 94.85%   |
| MSN-D1     | 93.39%   | Sncg      | 80.48%   |
| MSN-D2     | 91.40%   | ASC       | 95.68%   |
| Vip        | 94.00%   | L6b       | 93.41%   |
| Sst        | 94.19%   | L5-IT     | 94.60%   |
| Pvalb      | 92.62%   | Lamp5     | 94.74%   |
| L6-IT      | 84.32%   |           |          |

*Accuracy = Percentage of correctly predicted cells per type*

The high prediction accuracy across most cell types further demonstrates the strong performance of scWGBS-GPT in **single-cell methylation annotation tasks**.

## 📖 Citation
Please cite our paper if you use this code in your work:
```

@article{Liang2025.02.19.638959,

author = {Liang, Chaoqi and Ye, Peng and Yan, Hongliang and Zheng, Peng and Sun, Jianle and Wang, Yanni and Li, Yu and Ren, Yuchen and Jiang, Yuanpei and Xiang, Junjia and Zhang, Sizhe and Jiang, Linle and Bai, Weiqiang and Ma, Xinzhu and Chen, Tao and Zuo, Wangmeng and Bai, Lei and Ouyang, Wanli and Li, Jia},

title = {scWGBS-GPT: A Foundation Model for Capturing Long-Range CpG Dependencies in Single-Cell Whole-Genome Bisulfite Sequencing to Enhance Epigenetic Analysis},

year = {2025},

doi = {10.1101/2025.02.19.638959},

publisher = {Cold Spring Harbor Laboratory},

URL = {https://www.biorxiv.org/content/early/2025/02/23/2025.02.19.638959},

journal = {bioRxiv}

}

```

## 🚀 Quick Start

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/ChaoqiLiang/scWGBS-GPT-private-.git
cd scWGBS-GPT-private-
```

### 2. Requirements

- Python 3.9
- Required libraries:
  - `torch`
  - `mamba`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `pyfaidx`
  - `transformers`
  - `argparse`
  - `concurrent.futures`

To install these dependencies, you can run:

```bash
conda create -n scWGBS_GPT python=3.10 -y
conda activate scWGBS_GPT
pip install -r requirements.txt
```

### 3. Fine-Tuning for Colorectal Cancer Type Classification

This repository provides a framework for fine-tuning the scWGBS-GPT model, designed for processing single-cell Whole-Genome Bisulfite Sequencing (scWGBS) data. Specifically, it focuses on fine-tuning for colorectal cancer type classification using the scWGBS-GPT model. The model leverages transformer-based architectures with specialized attention mechanisms for sequence classification tasks.

#### Required Files
The fine-tuning dataset should be prepared with the following files:

- **scWGBS data** in `.npz` format, containing single-cell CpG site information.
- **CSV files** for training and testing sets, specifying cell metadata and corresponding labels.

You can download the required dataset from [Google Drive](https://drive.google.com/drive/folders/1PEAdRngaonY4TMEEX4tGO-zF7nqRuN52) and save it to the current directory (`./`). This will ensure that all necessary data files and model checkpoint are available for fine-tuning.

The paths to these files should be specified in the configuration file (`config/finetuning/colorectal_cancer_type/training_args_fp16.json`).

#### Path of Training Configuration:
Specify paths in `config/finetuning/colorectal_cancer_type/training_args_fp16.json`:

#### Configuration:
The following configuration files are key for fine-tuning:
- **`config/finetuning/colorectal_cancer_type/training_args_fp16.json`**: Contains training hyperparameters such as batch size, learning rate, and model paths.
- **`config/finetuning/colorectal_cancer_type/deepspeed_config_fp16.json`**: Optimizes training for large models using **DeepSpeed**.

#### Submit the SLURM job with:
```bash
sbatch script/finetuning/finetuning_colorectal_cancer_type.sh
```

### 4. Finetuning on Your Owner Data (scWGBS 6-mer Tokenizer and Processing Pipeline)

This repository provides a Python-based pipeline to process single-cell whole-genome bisulfite sequencing (scWGBS) data from `.tsv.gz` files. It extracts 6-mer nucleotide sequences around CpG sites, tokenizes them with a custom tokenizer, and outputs methylation ratios, positions, and chromosome identifiers.

#### What does this pipeline do?

1. **Searches for `.tsv.gz` files** in a specified directory (and its subdirectories).  
2. **Reads each `.tsv.gz` file** in chunks.  
3. **Extracts the 6-mer** around each CpG site.  
4. **Performs reverse-complement** for negative-strand reads.  
5. **Uses a custom tokenizer** (e.g., `scWGBS 6-mer tokenizer`) to convert the 6-mer sequence into integer token IDs.  
6. **Calculates methylation ratios** by merging coverage at the same positions (summing total reads and methylated reads).  
7. **Saves the token IDs**, methylation ratios, genomic positions, and chromosome identifiers in compressed `.npz` files.

This pipeline is designed for large-scale parallel processing of scWGBS data.

#### Features
- Processes `.tsv.gz` files containing scWGBS data.
- Extracts and tokenizes 6-mer sequences surrounding CpG sites.
- Computes methylation ratios and saves them in `.npz` files.
- Handles multiple chromosomes and large datasets using parallel processing.
- Efficient memory management by processing data in chunks.
- Uses a pre-trained tokenizer to convert sequences into token IDs.

#### Data Requirements

- **Reference Genome FASTA** (e.g., `mm10.fa` or `hg38.fa`) indexed via [pyfaidx](https://github.com/mdshw5/pyfaidx).  
  Make sure it’s accessible for random access. Typically, `pyfaidx` will create an `.fai` index the first time you access the file.  

- **scWGBS .tsv.gz files**: This pipeline assumes each row is of the form:

| Chromosome | Start Position | Strand | CpG Sequence | Methylated Reads | Total Reads |
|------------|----------------|--------|--------------|------------------|-------------|
| chr1       | 10000          | +      | ATCGAC       | 5                | 10          |

- For strand-specific data (`-` strand), you may need to shift the position or reverse complement the CpG sequence.

- **Strand `-`** implies you might shift or reverse complement when extracting the 6-mer.  

- **Chromosome List**: By default, the script includes `["chrM", "chrX", "chrY", "chr1", ..., "chr23"]`. Adjust this list as needed for your organism or reference genome.

#### Example Command

```bash
python data/process_scwgbs.py \
    --base_dir /path/to/scWGBS/data/your_tsv_path \
    --tokenizer_dir src/tokenizers/scwgbs_6mer \
    --chromosomes chrM chrX chrY chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 \
    --chunk_size 10000 \
    --max_workers 16
```

1. **`--base_dir`** is where the script will look for `.tsv.gz` files.  
3. **`--tokenizer_dir`** points to the custom tokenizer folder (which has `config.json` and `tokenizer.json`).  
4. **`--chromosomes`** example: only process `chrM, chrX, chrY, chr1, chr2, chr3` for testing.  
5. **`--chunk_size`** is 10000 lines at a time. Adjust for memory usage.  
6. **`--max_workers`** is how many parallel processes to spawn. Increase if you have more CPU cores and sufficient RAM.

#### Output Files

For each `.tsv.gz` file found, the script will generate four `.npz` files in new subdirectories (parallel to the directory containing the input file):

1. **`....npz`** (e.g., `sample1.npz`):  
   - Contains the array of token IDs for each 6-mer around valid CpG sites.

2. **`..._ratios.npz`** (e.g., `sample1_ratios.npz`):  
   - Contains the methylation ratios (methylated_reads / total_reads) per position.

3. **`..._positions.npz`** (e.g., `sample1_positions.npz`):  
   - Contains the genomic positions (1-based or 0-based depending on your handling).

4. **`..._chrs.npz`** (e.g., `sample1_chrs.npz`):  
   - Contains integer-coded chromosome identifiers. For example:
     - `chrM` -> -2  
     - `chrX` -> -1  
     - `chrY` -> 0  
     - `chr1` -> 1, etc.  

Directory structure example:

```
/home/.../GSE_data-mouse_processed/
├── <some_subdir>/
│   ├── sample1.tsv.gz
│   ├── sample2.tsv.gz
│   └── ...
│
├── scWGBS_with_nucleotide_change_6mer_tokens_npz/
│   └── <some_subdir>/
│       ├── sample1.npz
│       ├── sample2.npz
│       └── ...
├── scWGBS_with_nucleotide_change_6mer_ratios_npz/
│   └── <some_subdir>/
│       ├── sample1_ratios.npz
│       ├── sample2_ratios.npz
│       └── ...
├── scWGBS_with_nucleotide_change_6mer_positions_npz/
│   └── <some_subdir>/
│       ├── sample1_positions.npz
│       ├── sample2_positions.npz
│       └── ...
└── scWGBS_with_nucleotide_change_6mer_chrs_npz/
    └── <some_subdir>/
        ├── sample1_chrs.npz
        ├── sample2_chrs.npz
        └── ...
```

## ❓ FAQ

1. **Why do I see `[SKIP] ... already exists` in the output?**  
   The script checks if the `.npz` file (positions specifically) already exists to avoid reprocessing the same file. This is useful for resuming runs.

2. **Why is my code slow?**  
   - Large `.tsv.gz` files can be slow to read. Using the `chunk_size` parameter optimally can help.  
   - Increase `--max_workers` if you have more CPU cores.  

3. **Why do some `.npz` files have zero length?**  
   If an error occurs during processing, the script saves an empty `.npz` to mark incomplete data. Check the log for `[ERROR]` messages.

4. **How do I modify the script for a different organism (e.g., human `hg38`)?**  
   - Change `--reference_fasta` to the appropriate FASTA.  
   - Modify the chromosome list using `--chromosomes`.  

5. **Can I use this for single-end or paired-end data?**  
   - The script only depends on the `.tsv.gz` format (`chrom, pos, strand, seq, methylated, total`). It does not differentiate single-end vs. paired-end.  

## 📜 License

This project is licensed under the [MIT License](LICENSE). Please see the `LICENSE` file for details.

## 👥 Contributing

Contributions are welcome!  
- **Issues**: If you find a bug or have a feature request, open a [GitHub Issue](../../issues).  
- **Pull Requests**: Fork the repo, make changes, and create a PR.
