#!/usr/bin/env python3
"""
Script to split a large CSV file into multiple smaller chunks
without loading the entire file into memory at once.
"""

import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
import argparse


def get_file_line_count(file_path):
    """Count the number of lines in a file without loading it all at once."""
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in tqdm(file, desc="Counting lines"):
            line_count += 1
    return line_count

def split_csv(input_file, output_dir, num_chunks=20):
    """
    Split a CSV file into multiple chunks without loading the entire file at once.
    
    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Directory to save the chunks
        num_chunks (int): Number of chunks to create
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of lines in the file
    total_lines = get_file_line_count(input_file)
    print(f"Total lines in file: {total_lines}")
    
    # Calculate lines per chunk (including header in each chunk)
    header_line = None
    lines_per_chunk = (total_lines - 1) // num_chunks + 1  # -1 for header, +1 to include header in count
    
    # Open the input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Read the header
        header_line = infile.readline()
        
        chunk_num = 1
        line_count = 0
        outfile = None
        
        try:
            # Process the file line by line
            for line in infile:
                # If we need to start a new chunk
                if line_count % lines_per_chunk == 0:
                    # Close the previous chunk file if open
                    if outfile:
                        outfile.close()
                    
                    # Open a new chunk file
                    chunk_path = os.path.join(output_dir, f"bpmn_chunk_{chunk_num:02d}.csv")
                    outfile = open(chunk_path, 'w', encoding='utf-8')
                    
                    # Write the header to the new chunk
                    outfile.write(header_line)
                    
                    print(f"Creating chunk {chunk_num}: {chunk_path}")
                    chunk_num += 1
                
                # Write the current line to the chunk file
                outfile.write(line)
                line_count += 1
                
                # Print progress every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"Processed {line_count} lines...")
        
        finally:
            # Make sure to close the last file
            if outfile:
                outfile.close()
    
    print(f"Successfully split {input_file} into {chunk_num-1} chunks in {output_dir}")

if __name__ == "__main__":
    input_file = "datasets/bpmn2.0.csv"
    output_dir = "datasets/bpmn_chunks"
    parser = argparse.ArgumentParser(description="Split a large CSV file into smaller chunks.")
    parser.add_argument("--input_file", type=str, default=input_file, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save the chunks.")
    parser.add_argument("--num_chunks", type=int, default=20, help="Number of chunks to create.")
    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir
    num_chunks = args.num_chunks
    
    print(f"Splitting {input_file} into {num_chunks} chunks...")
    split_csv(input_file, output_dir, num_chunks)
