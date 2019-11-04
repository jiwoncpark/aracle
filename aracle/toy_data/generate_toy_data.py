# -*- coding: utf-8 -*-
"""Generating the toy data.

This script generates the toy data.

Example
-------
To run this script, pass in the desired config file as argument::
    $ generate_toy_data --n_data 1000

"""
import os, sys
import argparse
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from tqdm import tqdm
from aracle.toy_data import ToySquares

def parse_args():
    """Parse command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('n_data', type=int, help='number of X, Y data pairs to generate')
    parser.add_argument('t_offset', type=int, help='number of time steps between X and Y')
    parser.add_argument('dest_dir', type=str, help='destination directory')
    parser.add_argument('--canvas_size', default=224, dest='canvas_size', type=int,
                        help='size of the canvas on which the toy squares fall, in pixels (default: 224)')
    parser.add_argument('--n_objects', default=20, dest='n_objects', type=int,
                        help='number of objects to spawn (default: 20)')
    args = parser.parse_args()
    # sys.argv rerouting for setuptools entry point
    if args is None:
        args = SimpleNamespace()
        args.n_data = sys.argv[0]
        args.t_offset = sys.argv[1]
    return args

def get_max_time(canvas_size, t_offset):
    """Get the maximum number of reasonable time steps for a given ToySquares instance, until which not many objects will have fallen off the canvas

    Parameters
    ----------
    canvas_size : int
        size of the canvas on which the toy squares fall, in pixels
    t_offset : int
        number of time steps between X and Y

    """
    max_time = int(canvas_size/2)
    max_time = 2*(max_time//2) + 2 # round to nearest even number
    if (max_time == 1) or (max_time < t_offset):
        raise ValueError("Value supplied for t_offset is too high compared to the canvas size.")
    return max_time

def main():
    args = parse_args()
    max_time = get_max_time(args.canvas_size, args.t_offset)
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
        print("Saving npy files and metadata at {:s}".format(str(os.path.abspath(args.dest_dir))))
    else:
        raise OSError("Destination folder already exists.")

    current_n = 0 # running idx of image
    n_data_per_toy_square = max_time - args.t_offset
    n_toy_squares_minus_1, n_data_for_last_toy_square = divmod(args.n_data, n_data_per_toy_square)
    n_time_steps_for_last_toy_square =  n_data_for_last_toy_square + args.t_offset if n_data_for_last_toy_square else 0
    n_img = max_time*n_toy_squares_minus_1 + n_time_steps_for_last_toy_square*n_data_for_last_toy_square
    print(n_img, n_toy_squares_minus_1, max_time, args.t_offset, n_data_per_toy_square, n_time_steps_for_last_toy_square)
    pbar = tqdm(total=n_img) 
    metadata_df = pd.DataFrame(columns=['img_idx', 'img_filename', 'toy_squares_id', 'time'])
    n_time_steps = max_time
    for toy_squares_id in range(n_toy_squares_minus_1 + 1):
        toy_squares = ToySquares(args.canvas_size, args.n_objects)
        if toy_squares_id == n_toy_squares_minus_1: # last toy_squares
            n_time_steps = n_time_steps_for_last_toy_square
        for t in range(n_time_steps):
            img_filename = 'toy_{0:07d}.npy'.format(current_n)
            #toy_squares.export_image(img_filename)
            toy_squares.increment_time_step()
            metadata_df = metadata_df.append(
                                             dict(img_idx=current_n,
                                                  img_filename=img_filename,
                                                  time=t,
                                                  toy_squares_id=toy_squares_id),
                                             ignore_index=True,
                                             )
            current_n += 1
            pbar.update(1)
        del toy_squares # FIXME: bad practice and probably not necessary
    # Export command-line arguments
    with open(os.path.join(args.dest_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Export metadata csv file
    metadata_df.to_csv(os.path.join(args.dest_dir, 'metadata.csv'), index=None)
    pbar.close()


if __name__ == '__main__':
    main()