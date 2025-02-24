#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2025 ChaoqiLiang

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
from transformers import AutoTokenizer

def find_tsv_files(base_dir: str) -> list:
    """
    Recursively find all .tsv.gz files under a base directory.
    """
    tsv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.tsv.gz'):
                tsv_files.append(os.path.join(root, file))
    return tsv_files

def process_file(args_tuple):
    """
    Process a single .tsv.gz file to extract 6-mer token IDs, 
    methylation ratios, positions, and chromosome identifiers.
    
    :param args_tuple: A tuple containing (file_path, tokenizer, chromosomes, chunk_size) used for parallelization.
    :return: None. Results are saved as .npz files.
    """
    file_path, tokenizer, chromosomes, chunk_size = args_tuple
    
    # Prepare output directory structure
    grandparent_dir = os.path.dirname(os.path.dirname(file_path))
    second_level_dir = os.path.basename(os.path.dirname(file_path))
    
    output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_6mer_tokens_npz')
    ratios_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_6mer_ratios_npz')
    positions_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_6mer_positions_npz')
    chrs_output_base_dir = os.path.join(grandparent_dir, 'scWGBS_with_nucleotide_change_6mer_chrs_npz')
    
    output_dir = os.path.join(output_base_dir, second_level_dir)
    ratios_output_dir = os.path.join(ratios_output_base_dir, second_level_dir)
    positions_output_dir = os.path.join(positions_output_base_dir, second_level_dir)
    chrs_output_dir = os.path.join(chrs_output_base_dir, second_level_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ratios_output_dir, exist_ok=True)
    os.makedirs(positions_output_dir, exist_ok=True)
    os.makedirs(chrs_output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.tsv.gz', '.npz'))
    ratios_output_file = os.path.join(ratios_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_ratios.npz'))
    positions_output_file = os.path.join(positions_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_positions.npz'))
    chrs_output_file = os.path.join(chrs_output_dir, os.path.basename(file_path).replace('.tsv.gz', '_chrs.npz'))
    
    if os.path.exists(positions_output_file):
        print(f"[SKIP] {positions_output_file} already exists.")
        return
    
    np.savez_compressed(positions_output_file, data=np.array([]))
    print(f"[PROCESSING] {file_path}")
    
    try:
        all_tokenized_sequences = []
        all_mets_sequences = []
        all_totals_sequences = []
        all_positions_sequences = []
        all_chrs_sequences = []
        
        for chrom in chromosomes:
            pos_index_dict = {}
            
            for chunk in pd.read_csv(file_path, sep='\t', header=None, chunksize=chunk_size):
                chrom_df = chunk[chunk.iloc[:, 0] == chrom]

                if chrom_df.empty:
                    continue

                positions = chrom_df.iloc[:, 1].astype(int).values
                strands = chrom_df.iloc[:, 2].values
                mets = chrom_df.iloc[:, 4].astype(float).values
                totals = chrom_df.iloc[:, 5].astype(float).values
                CpG_seqs = chrom_df.iloc[:, 3].astype(str).values

                for i in range(len(chrom_df)):
                    position = positions[i] - 1 if strands[i] == '-' else positions[i]
                    seq_6mer = CpG_seqs[i]
                    
                    if seq_6mer[4:6] == "CG":
                        if position not in pos_index_dict:
                            pos_index_dict[position] = len(all_positions_sequences)

                            tokenized_sequence = tokenizer.convert_tokens_to_ids(seq_6mer)
                            all_tokenized_sequences.append(tokenized_sequence)
                            all_mets_sequences.append(mets[i])
                            all_totals_sequences.append(totals[i])
                            all_positions_sequences.append(position)
                            
                            chr_id = chrom.replace("chr", "")
                            chr_id = -2 if chr_id == "M" else -1 if chr_id == "X" else 0 if chr_id == "Y" else int(chr_id) if chr_id.isdigit() else 99
                            all_chrs_sequences.append(chr_id)
                        else:
                            idx = pos_index_dict[position]
                            all_mets_sequences[idx] += mets[i]
                            all_totals_sequences[idx] += totals[i]
        
        token_ids_array = np.array(all_tokenized_sequences, dtype=np.int32)
        mets_array = np.array(all_mets_sequences, dtype=np.float32)
        totals_array = np.array(all_totals_sequences, dtype=np.float32)
        ratios_array = mets_array / totals_array
        positions_array = np.array(all_positions_sequences, dtype=np.int32)
        chrs_array = np.array(all_chrs_sequences, dtype=np.int32)
        
        np.savez_compressed(output_file, data=token_ids_array)
        np.savez_compressed(ratios_output_file, data=ratios_array)
        np.savez_compressed(positions_output_file, data=positions_array)
        np.savez_compressed(chrs_output_file, data=chrs_array)
        
        print(f"[DONE] Saved results for {file_path}")
    except Exception as e:
        print(f"[ERROR] Processing {file_path}: {e}")
        np.savez_compressed(positions_output_file, data=np.array([]))

def main():
    """
    Main function to handle argument parsing, file finding, and parallel processing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process scWGBS .tsv.gz files to extract 6-mer token IDs, "
            "methylation ratios, positions, and chromosomes, saving them as .npz files."
        )
    )
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing subfolders with .tsv.gz files.")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory containing the scWGBS 6-mer tokenizer.")
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=["chrM", "chrX", "chrY", "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chr23"],
        help="List of chromosomes to process."
    )
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for reading large .tsv.gz files.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel processes to use.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    file_list = find_tsv_files(args.base_dir)
    print(f"Found {len(file_list)} .tsv.gz files under {args.base_dir}.")

    task_args = [(f, tokenizer, args.chromosomes, args.chunk_size) for f in file_list]

    print(f"Using ProcessPoolExecutor with max_workers={args.max_workers}")
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(process_file, task_args), total=len(task_args)))

    print("[ALL DONE] All files processed.")

if __name__ == "__main__":
    main()
