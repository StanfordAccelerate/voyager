import argparse
import dill as pickle
import numpy as np
import torch
import re
import sys, getopt
import os
import struct

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
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Input pickle datafiles.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output datafiles.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="Number of training steps of datafiles.",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.datapath, "datafile") if args.output_dir is None else args.output_dir
    for i in range(args.n_steps):
        dataDir = os.path.join(output_dir, f"step{i}")
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        files = ["activations", "errors", "weights", "gradients"]
        for filename in files:
            path = os.path.join(args.datapath, "step_" + str(i) + "_" + filename + ".pkl")
            print("Reading input file: " + path)

            subDir = os.path.join(dataDir, filename)
            if not os.path.exists(subDir):
                os.makedirs(subDir)

            with open(path, 'rb') as infile:
                parameters = pickle.load(infile)

            for name, param in parameters.items():
                outfile = os.path.join(subDir, name.replace('.', '_'))
                write_fp64(outfile, param)
                print(outfile, end="\r", flush=True)
            print("=" * 120)
