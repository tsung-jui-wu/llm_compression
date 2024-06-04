import os
import argparse
import pathlib
import torchaudio

from tqdm import tqdm
from torchaudio.transforms import Resample

def get_config():
    parser = argparse.ArgumentParser(
        prog='Chunk Librispeech Dataset',
        description='This program takes librispeech dataset, and chunks audio into preferable content length (defaut 1024) per file',
    )
    
    parser.add_argument("--input_dir", type=pathlib.Path, help="dir of downloaded file from Librispeech")
    parser.add_argument("--output_dir", type=pathlib.Path, help="dir of wav you want to output")
    parser.add_argument("--len", default=1024, type=int, help="content length (default 1024)")

    return parser


def main():
    args = get_config()

        

if __name__ == '__main__':
    main()