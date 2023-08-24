#!/usr/bin/env python3

import argparse
import dill as pickle
import struct
import os

import numpy as np


''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    data = data.astype(np.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)

def dump_pickled_data(path, output_dir):
    with open(path,  'rb') as f:
        data = pickle.load(f)
    
    # get parent directory name
    parent_dir = os.path.basename(os.path.dirname(path))
    # get name of file without extension
    filename = os.path.basename(path).split('.')[0]

    output_dir = os.path.join(output_dir, parent_dir, filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, param in data.items():
        outfile = os.path.join(output_dir, name.replace('.', '_'))
        write_fp64(outfile, param.flatten())
        print(outfile, end="\r", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Dump training data for MobileBERT")
    parser.add_argument("--data_dir", type=str, help="Path to data directory.", default=None, required=True)
    parser.add_argument("--output_dir", type=str, help="Path to output directory.", default=None, required=True)
    args = parser.parse_args()

    # iterate over all folders in data_dir
    for folder in os.listdir(args.data_dir):
        # iterate over all files in folder
        for file in os.listdir(os.path.join(args.data_dir, folder)):
            # only parse pickle files
            if file.endswith('.pkl'):
                print("Parsing " + os.path.join(args.data_dir, folder, file))
                dump_pickled_data(os.path.join(args.data_dir, folder, file), args.output_dir)

if __name__ == "__main__":
    main()