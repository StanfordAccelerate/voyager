#!/usr/bin/env python3

import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import argparse
import functools
import pickle
import struct
import os
import re

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Local modules
import resnet_reference
import resnet_record
import bn_folding

# TODO(fpedd): Re-implement the int8 part (what is that used for, actually?)
# TODO(fpedd): What are the datatypes supposed to mean, anyway?
# TODO(fpedd): Switch over to generating data for multiple images/inferences


def prepare_data(args: argparse.Namespace):

    # Walk the category directories and collect the image paths
    image_paths = []
    for dir, _, files in os.walk(args.data_dir):
        for f in files:
            image_paths.append(os.path.join(dir, f))

    # Get (all) reference labels from the labels file in the root of the data directory
    ref_labels = []
    with open(image_paths[0], "r") as f:
        ref_labels = [s.strip().split(',') for s in f.readlines()]

    # Remove label file from path list
    image_paths = image_paths[1:]

    # Get label to every image by extracting it from the path
    image_labels = [re.findall('n[0-9]+', i)[0] for i in image_paths]

    return image_paths, image_labels, ref_labels


def run_model(args: argparse.Namespace, image_paths: str, image_labels: str, ref_labels: str):

    if args.use_model_url:
        print("Downloading model from {}".format(args.model_url))
        state_dict = load_state_dict_from_url(args.model_url)
    else:
        # Alternatively, use torch.load and load weights from file in model_dir
        model_path = os.path.join(args.model_dir, 'resnet18-5c106cde.pth')
        print("Loading model from disk {}".format(model_path))
        state_dict = torch.load(model_path)  # .state_dict()

    # Force PyTorch to CPU
    torch.device("cpu")

    # Create reference model and fill with weights
    model_ref = resnet_reference.ResNet(
        block=resnet_reference.BasicBlock, layers=[2, 2, 2, 2])
    model_ref.load_state_dict(state_dict)
    model_ref.eval()

    # Create bn folded model (bn folded into conv)
    model_bnfold = resnet_record.ResNet(
        block=resnet_record.BasicBlock, layers=[2, 2, 2, 2])
    model_bnfold.load_state_dict(state_dict)
    model_bnfold.eval()
    model_bnfold = bn_folding.bn_folding_model(model_bnfold)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Loop over all images and verify results
    for i, image_path in enumerate(image_paths):

        # Load and prepare input image
        input_image = Image.open(image_path)
        input = preprocess(input_image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output_ref = model_ref(input)
            output_bnfold = model_bnfold(input)

        # Get output probabilities
        output_probs_ref = torch.nn.functional.softmax(
            output_ref.squeeze(), dim=0)
        output_probs_bnfold = torch.nn.functional.softmax(
            output_bnfold.squeeze(), dim=0)

        # Get top probability and category id
        top_prob_ref, top_catid_ref = torch.topk(output_probs_ref, 5)
        top_prob_bnfold, top_catid_bnfold = torch.topk(output_probs_bnfold, 5)

        # Compare output between models
        diff = functools.reduce(
            (lambda acc, top: acc or top[0] != top[1]), zip(top_catid_ref, top_catid_bnfold), False)

        if diff:
            raise ValueError("Top 5 differ at {} with {} vs {}".format(
                image_labels[i], top_catid_ref, top_catid_bnfold))

        # Compare model top category to reference category
        # TODO(fpedd): ResNet is not perfect, so we ignore differences to reference for now
        #              We are happy as long as the two model match
        # if image_labels[i] != ref_labels[top_catid_ref[0]][0]:
        #     raise ValueError("Top 1 differs to refernce {} vs {} ({})".format(
        #         image_labels[i], ref_labels[top_catid_ref[0]][0], ref_labels[top_catid_ref[0]][1]))

        if resnet_record._export:

            # Get unique name from image path
            model_id = re.findall('000[0-9]+', image_path)[-1]

            os.makedirs(args.model_dir, exist_ok=True)
            pkl_file_path = os.path.join(args.model_dir, model_id + ".pkl")
            with open(pkl_file_path, "wb") as f:
                pickle.dump(resnet_record._buffer, f)

            # TODO(fpedd): We only want to record and save the first model instance (for now)
            resnet_record._export = False

    print("\nRan through {} samples from {} categories to verify model.".format(
        len(image_paths), len(set(image_labels))))

    return pkl_file_path


def write_binary(args: argparse.Namespace, pkl_file_path: str):
    print("Writing binary files to: {}".format(args.binary_dir))

    # Create output_dir if it does not already exists
    os.makedirs(args.binary_dir, exist_ok=True)

    # Load pickle file and write binary layer files
    with open(pkl_file_path, 'rb') as input:

        # Working with posit8 datatypes
        if args.data_type == 'posit8':

            # Iterate over all entries in the pickled dictionary
            for name, data in pickle.load(input).items():

                # Convert name to filename (special case for downsample)
                file_name = re.sub('downsample.0', 'downsample', name)
                file_name = re.sub('\.', '_', file_name)

                # Open output file, then pack and write data
                with open(os.path.join(args.binary_dir, file_name), 'wb') as output:
                    output.write(struct.pack('%sd' % len(data), *data))

        elif args.data_type == 'int8':
            raise NotImplementedError("Currently int8 is not supported!")

        else:
            raise ValueError("Unsupported datatype {}!".format(args.data_type))
    return


def main():

    parser = argparse.ArgumentParser(
        description='ResNet data generator (from model and images to binary data).')
    parser.add_argument('-ddir', '--data_dir',
                        type=str,
                        default='../../data/imagenette',
                        help='Path to dataset.')
    parser.add_argument('-url', '--model_url',
                        type=str,
                        default='https://download.pytorch.org/models/resnet18-5c106cde.pth',
                        help='URL pointing to a state_dict (.pth ot .pt).')
    parser.add_argument('-umu', '--use_model_url',
                        default=True,
                        action='store_false',
                        help='Download model data from URL. If false, use local model file.')
    parser.add_argument('-mdir', '--model_dir',
                        type=str,
                        default='model_data',
                        help='Path to model directory the .pkl intermediate file will be written to. If  (.pth ot .pt).')
    parser.add_argument('-bdir', '--binary_dir',
                        type=str,
                        default='binary_data',
                        help='Path the binary data output will be written to (binary files for simulation).')
    parser.add_argument('-t', '--data_type',
                        type=str,
                        default='posit8', help='Datatype to parse (posit8, int8).')
    args = parser.parse_args()

    # Convert paths from relative to absolute (allows calling script from different dir)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    args.data_dir = os.path.join(script_dir, args.data_dir)
    args.model_dir = os.path.join(script_dir, args.model_dir)
    args.binary_dir = os.path.join(script_dir, args.binary_dir)

    # Run through three functions
    image_paths, image_labels, ref_labels = prepare_data(args)
    pkl_file_path = run_model(args, image_paths, image_labels, ref_labels)
    write_binary(args, pkl_file_path)

    print("ResNet data generator done!")


if __name__ == "__main__":
    main()
