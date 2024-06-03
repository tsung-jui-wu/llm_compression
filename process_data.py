import os
import torch
from torch.utils.data import Dataset

class BinaryDataset(Dataset):
    def __init__(self, args):

        print("\nloading dataset......")
        
        self.directories = args.dataset_path
        self.transform = ToBinaryString(args.bits, args.seq_len)

        self.filenames = []
        for directory in self.directories:
            full_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                          if any(f.endswith(filetype) for filetype in args.modalities)]
            self.filenames.extend(full_paths)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Construct the full path to the file
        path = self.filenames[idx]
        
        # Read the file as bytes
        with open(path, 'rb') as file:
            b = file.read()

        if self.transform:
            b = self.transform(b)
        
        return b, self.filenames[idx]

class ToBinaryString:
    def __init__(self, bits=8, segment_length=256):
        '''
        Args:
            bits (Int): Number of bits to group together into an integer
            chunk_size (Int): Number of integers (tokens) per segment
        '''
        self.bits = bits
        self.chunk_size = segment_length

    def __call__(self, input_bytes):
        # Convert bytes to binary string

        binary_string = ''.join(f'{byte:08b}' for byte in input_bytes)

        integers = [int(binary_string[i:i+self.bits], 2) for i in range(0, len(binary_string), self.bits)]
        tensor = torch.tensor(integers)

        padding_size = (self.chunk_size - tensor.size(0) % self.chunk_size) % self.chunk_size

        # Pad the tensor if necessary
        if padding_size > 0:
            tensor = torch.cat([tensor, torch.zeros(padding_size, dtype=tensor.dtype)])

        total_length = tensor.size(0) + padding_size
        tensor = tensor.view(-1, self.chunk_size)

        return tensor
