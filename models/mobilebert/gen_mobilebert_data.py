#!/usr/bin/env python3
import argparse
import os
from os import listdir
import struct
import sys
import re

import dill as pickle
import numpy as np
import torch


''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    data = data.astype(np.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold model datafile from pickle files.")
    parser.add_argument("datapath", type=str, help="Input pickle datafiles.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output datafiles.",
    )
    args = parser.parse_args()

    dataDir = os.path.join(args.datapath, "datafile") if args.output_dir is None else args.output_dir
    files = [f for f in listdir(args.datapath) if f.endswith('.pkl')]
    print(files)
    for filename in files:
        path = os.path.join(args.datapath, filename)
        print("Parsing " + path)

        subDir = os.path.join(dataDir, filename[:-4])
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        with open(path, 'rb') as infile:
            parameters = pickle.load(infile)

        for name, param in parameters.items():
            outfile = os.path.join(subDir, name.replace('.', '_'))
            write_fp64(outfile, param.flatten())
            print(outfile, end="\r", flush=True)
        print("=" * 120)
