#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) <YEAR> <YOUR NAME>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in the 
Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the 
following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pyfaidx import Fasta
from transformers import AutoTokenizer

__author__ = "Your Name"
__version__ = "1.0.0"

def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    
    :param sequence: Original DNA sequence.
    :return: Reverse complement DNA sequence.
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    complement_seq = "".join(complement.get(base, "N") for base in sequence)
    return complement_seq[::-1]


def find_tsv_files(base_dir: str) -> list:
    """
    Recursively find all .tsv.gz files under a base directory.
    
    :param base_dir: The directory to search.
    :return: A list of file paths ending with .tsv.gz.
    """
    tsv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tsv.gz'):
                tsv_files.append(os.path.join(root, file))
    return tsv_files


def process_file(args_tuple):
    """
    Process a single .tsv.gz file to extract 10-mer token IDs, 
    methylation ratios, positions, and chromosome identifiers.
    
    The results are saved in corresponding .npz files to 
    avoid reprocessing the same data.
    
    :param args_tuple: A tuple containing (file_path, tokenizer, hg38_fasta_path, 
                      chromosomes, chunk_size, max_workers) used for parallelization.
    :return: None. Results are saved as .npz files.
    """
    file_path, tokenizer, fasta_path, chromosomes, chunk_size = args_tuple
    
    # Prepare output directory structure
    grandparent_dir = os.path.dirname(os.path.dirname(file_path))
    second_level_dir = os.path.basename(os.path.dirname(file_path))
    
    output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_10mer_tokens_npz')
    ratios_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_10mer_ratios_npz')
    positions_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_10mer_positions_npz')
    chrs_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_10mer_chrs_npz')
    
    output_dir = os.path.join(output_base_dir, second_level_dir)
    ratios_output_dir = os.path.join(ratios_output_base_dir, second_level_dir)
    positions_output_dir = os.path.join(positions_output_base_dir, second_level_dir)
    chrs_output_dir = os.path.join(chrs_output_base_dir, second_level_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ratios_output_dir, exist_ok=True)
    os.makedirs(positions_output_dir, exist_ok=True)
    os.makedirs(chrs_output_dir, exist_ok=True)

    # Output file paths
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.tsv.gz', '.npz'))
    ratios_output_file = os.path.join(ratios_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_ratios.npz'))
    positions_output_file = os.path.join(positions_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_positions.npz'))
    chrs_output_file = os.path.join(chrs_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_chrs.npz'))
    
    # Skip if the positions file already exists
    if os.path.exists(positions_output_file):
        print(f"[SKIP] {positions_output_file} already exists.")
        return
    
    # Create an empty file to signal "in process" (helps with concurrency checks)
    np.savez_compressed(positions_output_file, data=np.array([]))
    print(f"[PROCESSING] {file_path}")
    
    try:
        # Load reference genome in this process
        fasta_reference = Fasta(fasta_path)

        all_tokenized_sequences = []
        all_mets_sequences = []
        all_totals_sequences = []
        all_positions_sequences = []
        all_chrs_sequences = []
        
        # We'll keep a dictionary to track positions we've already processed
        # for the current file (merging coverage for duplicates).
        for chrom in chromosomes:
            # Dictionary to map positions -> index in all_positions_sequences
            pos_index_dict = {}
            
            # Read in chunks to limit memory usage
            for chunk in pd.read_csv(file_path, sep='\t', header=None, chunksize=chunk_size):
                chrom_df = chunk[chunk.iloc[:, 0] == chrom]

                if chrom_df.empty:
                    continue

                positions = chrom_df.iloc[:, 1].astype(int).values
                strands   = chrom_df.iloc[:, 2].values
                mets      = chrom_df.iloc[:, 4].astype(float).values
                totals    = chrom_df.iloc[:, 5].astype(float).values
                CpG_seqs  = chrom_df.iloc[:, 3].astype(str).values

                for i in range(len(chrom_df)):
                    # If strand is '-', we shift position by one
                    if strands[i] == '-':
                        position = positions[i] - 1
                    else:
                        position = positions[i]
                    
                    # Retrieve a 10-mer around the position
                    # (position-5:position+5) - using 0-based indexing
                    fasta_seq = fasta_reference[chrom][(position-5):(position+5)].seq.upper()
                    
                    # Possibly reverse complement the CpG context
                    cpg_seq = CpG_seqs[i]
                    if strands[i] == '-':
                        cpg_seq = reverse_complement(cpg_seq)
                        # Insert the cpg_seq into the 10-mer
                        #   first 3 nucleotides + cpg_seq + the remaining 3 nucleotides
                        seq_10mer = fasta_seq[:3] + cpg_seq + fasta_seq[6:]
                    else:
                        # Insert the cpg_seq at the center
                        seq_10mer = fasta_seq[:4] + cpg_seq + fasta_seq[7:]
                    
                    # Only keep if the middle is "CG"
                    if seq_10mer[4:6] == "CG":
                        # Check if we've seen this position before
                        if position not in pos_index_dict:
                            # Mark this new position
                            pos_index_dict[position] = len(all_positions_sequences)

                            tokenized_sequence = tokenizer.convert_tokens_to_ids(seq_10mer)
                            all_tokenized_sequences.append(tokenized_sequence)
                            all_mets_sequences.append(mets[i])
                            all_totals_sequences.append(totals[i])
                            all_positions_sequences.append(position)
                            
                            # Convert chromosome names to int (example: chrM -> -2, chrX -> -1, chrY -> 0, chr1->1, etc.)
                            # Adjust according to your specific mapping needs.
                            chr_id = chrom.replace("chr", "")
                            if chr_id == "M":
                                chr_id = -2
                            elif chr_id == "X":
                                chr_id = -1
                            elif chr_id == "Y":
                                chr_id = 0
                            else:
                                try:
                                    chr_id = int(chr_id)
                                except ValueError:
                                    # If it's something unrecognized, set to large number
                                    chr_id = 99  
                            all_chrs_sequences.append(chr_id)
                        else:
                            # If the position is seen before, add coverage
                            idx = pos_index_dict[position]
                            all_mets_sequences[idx] += mets[i]
                            all_totals_sequences[idx] += totals[i]
        
        # Convert to numpy arrays
        token_ids_array = np.array(all_tokenized_sequences, dtype=np.int32)
        mets_array      = np.array(all_mets_sequences,      dtype=np.float32)
        totals_array    = np.array(all_totals_sequences,    dtype=np.float32)
        ratios_array    = mets_array / totals_array
        positions_array = np.array(all_positions_sequences, dtype=np.int32)
        chrs_array      = np.array(all_chrs_sequences,      dtype=np.int32)
        
        # Save
        np.savez_compressed(output_file, data=token_ids_array)
        np.savez_compressed(ratios_output_file, data=ratios_array)
        np.savez_compressed(positions_output_file, data=positions_array)
        np.savez_compressed(chrs_output_file, data=chrs_array)
        
        print(f"[DONE] Saved results for {file_path}")
    except Exception as e:
        print(f"[ERROR] Processing {file_path}: {e}")
        # If error, ensure positions file is empty to mark incomplete
        np.savez_compressed(positions_output_file, data=np.array([]))


def main():
    """
    Main function to handle argument parsing, file finding, and parallel processing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process scWGBS .tsv.gz files to extract 10-mer token IDs, "
            "methylation ratios, positions, and chromosomes, saving them as .npz files."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing subfolders with .tsv.gz files."
    )
    parser.add_argument(
        "--reference_fasta",
        type=str,
        required=True,
        help="Path to reference genome FASTA file (e.g., mm10.fa or hg38.fa)."
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Directory containing the scWGBS 10-mer tokenizer."
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=[
            "chrM", "chrX", "chrY", "chr1", "chr2", "chr3", "chr4", "chr5", 
            "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", 
            "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", 
            "chr20", "chr21", "chr22", "chr23"
        ],
        help="List of chromosomes to process."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Chunk size for reading large .tsv.gz files."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel processes to use."
    )
    args = parser.parse_args()

    # Load the tokenizer once in the main process
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Find all .tsv.gz files
    file_list = find_tsv_files(args.base_dir)
    print(f"Found {len(file_list)} .tsv.gz files under {args.base_dir}.")

    # Prepare argument tuples for each file
    task_args = [
        (f, tokenizer, args.reference_fasta, args.chromosomes, args.chunk_size)
        for f in file_list
    ]

    # Parallel processing
    print(f"Using ProcessPoolExecutor with max_workers={args.max_workers}")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(process_file, task_args), total=len(task_args)))

    print("[ALL DONE] All files processed.")


if __name__ == "__main__":
    main()
