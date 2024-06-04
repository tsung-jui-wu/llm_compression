import os
import argparse
import pathlib
import numpy as np

from tqdm import tqdm
from PIL import Image


def get_config():
    parser = argparse.ArgumentParser(
        prog='Chunk Imagenet Dataset',
        description='This program takes imagenet_npz, resizes it to given size (defaut 32*32) and converts to grayscale',
    )
    
    parser.add_argument("--input_dir", type=pathlib.Path, help="dir of downloaded npz from Imagenet")
    parser.add_argument("--output_dir", type=pathlib.Path, help="dir of images you want to output")
    parser.add_argument("--image_size", default=32, type=int, help="image size (n*n), default is 32 (32*32)")

    return parser


def main():
    args = get_config()
    data = np.load(args.input_dir)
    images = data['data']
    
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in tqdm(images.shape[0]):
        image = images[i].reshape(3, args.image_size, args.image_size)
        image = image.transpose((1,2,0))

        img = Image.fromarray(image.astype('uint8'), 'RGB')
        img = img.convert('L')

        path_png = os.path.join(output_dir, f'{i}.png')

        img.save(path_png, format='PNG', compress_level=0)
        

if __name__ == '__main__':
    main()