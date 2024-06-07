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

def chunk_and_convert_flac_to_wav(input_dir, output_dir, target_sample_rate=16000, chunk_duration_seconds=1):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all FLAC files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.flac'):
            input_path = os.path.join(input_dir, filename)
            
            # Load the FLAC file
            waveform, sample_rate = torchaudio.load(input_path, format='flac')
            
            # Resample if necessary
            if sample_rate != target_sample_rate:
                resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = target_sample_rate
            
            # Calculate the number of samples for the chunk duration
            num_samples_per_chunk = sample_rate * chunk_duration_seconds
            
            # Process and save each chunk
            total_samples = waveform.size(1)
            for i in range(0, total_samples, num_samples_per_chunk):
                end = i + num_samples_per_chunk
                if end <= total_samples:
                    chunk = waveform[:, i:end]
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{i // num_samples_per_chunk}.wav"
                    output_path = os.path.join(output_dir, chunk_filename)
                    torchaudio.save(output_path, chunk, sample_rate)
                else:
                    print(f"Skipping chunk from {i} to {end} (less than {chunk_duration_seconds} second)")

def main():
    args = get_config()
    chunk_and_convert_flac_to_wav(args.input_dir, args.output_dir)
        

if __name__ == '__main__':
    main()