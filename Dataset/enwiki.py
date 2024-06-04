import re
import os
import argparse
import pathlib

from tqdm import tqdm

def read_and_split_file(input_path, output_dir, chunk_size=1024):
    # Read the file
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into chunks of 1024 characters
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    os.makedirs(output_dir, exists_ok=True)
    # Save each chunk to a new file
    for idx, chunk in tqdm(enumerate(chunks)):
        chunk_file_path = os.path.join(output_dir, f"chunk_{idx + 1}.txt")
        with open(chunk_file_path, 'w', encoding='ascii', errors='replace') as chunk_file:
            chunk_file.write(chunk)


def get_config():
    parser = argparse.ArgumentParser(
        prog='Chunk Librispeech Dataset',
        description='This program takes enwiki dataset, and chunks text into preferable content length (defaut 1024) per file',
    )
    
    parser.add_argument("--input_dir", type=pathlib.Path, help="dir of downloaded file from enwiki")
    parser.add_argument("--output_dir", type=pathlib.Path, help="dir of wav you want to output")
    parser.add_argument("--len", default=1024, type=int, help="content length (default 1024)")

    return parser

def main():
    args = get_config()
    read_and_split_file(
        input_path=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.len
    )   

if __name__ == '__main__':
    main()