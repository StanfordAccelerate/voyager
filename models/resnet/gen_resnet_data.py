#!/usr/bin/env python3
import argparse
import functools
import os
import pickle
import re
import struct

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm

import torchvision.models as models
from torchvision import transforms


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Local modules
import resnet_reference
import vision_models
import bn_folding

from vision_models import arrange_data

# TODO(fpedd): Re-implement the int8 part (what is that used for, actually?)
# TODO(fpedd): What are the datatypes supposed to mean, anyway?
# TODO(fpedd): Switch over to generating data for multiple images/inferences


def prepare_data(args: argparse.Namespace):
    # Walk the category directories and collect the image paths
    image_paths = []
    for dir, _, files in os.walk(args.data_dir):
        for f in files:
            image_paths.append(os.path.join(dir, f))
    
    # Only make single sample case deterministic (as per Kartik's request)
    if args.samples == 1:
        image_paths = sorted(image_paths)

    # Get (all) reference labels from the labels file in the root of the data directory
    ref_labels = []
    with open(image_paths[0], "r") as f:
        ref_labels = [s.strip().split(',') for s in f.readlines()]

    # Remove label file from path list
    image_paths = image_paths[1:]

    # Get label to every image by extracting it from the path
    image_labels = [re.findall('n[0-9]+', i)[0] for i in image_paths]

    return image_paths, image_labels, ref_labels


def dump_model_data(args: argparse.Namespace, image_paths: str, image_labels: str, ref_labels: str):
    model = vision_models.ResNet(block=vision_models.BasicBlock, layers=[2, 2, 2, 2])

    # Load default model
    model_path = os.path.join(args.model_dir, 'resnet18_mp2.pth')
    state_dict = torch.load(model_path).state_dict()
    model.load_state_dict(state_dict)

    model.eval()
    model = bn_folding.bn_folding_model(model)

    # Load QAT model
    ckpt_file = os.path.join(args.model_dir, args.model_file)
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    activations = {}

    def get_forward_pre_hook(name):
        def hook_fn(m, inputs):
            # print(f"{name}.input")
            activations[f"{name}.input"] = arrange_data(name, inputs[0])
        return hook_fn
    
    def get_forward_hook(name):
        def hook_fn(m, inputs, output):
            # print(f"{name}.comp", type(m).__name__)
            activations[f"{name}.comp"] = arrange_data(name, output)
        return hook_fn

    conv_layer = None
    handle = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            mod.register_forward_pre_hook(get_forward_pre_hook(name))
            if "downsample" in name:
                mod.register_forward_hook(get_forward_hook(name))
            else:
                conv_layer = name
                handle = mod.register_forward_hook(get_forward_hook(name))

        if isinstance(mod, (nn.MaxPool2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
            if handle:
                handle.remove()
            handle = mod.register_forward_hook(get_forward_hook(conv_layer))

        if isinstance(mod, torch.nn.Linear):
            mod.register_forward_pre_hook(get_forward_pre_hook(name))
            mod.register_forward_hook(get_forward_hook(name))

    for name, param in model.named_parameters():
        activations[name] = arrange_data(name, param.data)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model_ref_corr = 0
    pkl_file_paths = []

    # Loop over all images and verify results
    for i in tqdm(range(args.samples)):

        image_path = image_paths[i]
        print(image_path)

        # Load and prepare input image
        input_image = Image.open(image_path).convert('RGB')
        input = preprocess(input_image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(input)

        # Get output probabilities
        output_probs = torch.nn.functional.softmax(output.squeeze(), dim=0)
        top_prob_ref, top_catid_ref = torch.topk(output_probs, 5)

        if image_labels[i] == ref_labels[top_catid_ref[0]][0]:
            model_ref_corr += 1

        # Get unique name from image path
        path_components = os.path.normpath(image_path).split(os.path.sep)
        model_id = path_components[-2] + '_' + \
            re.findall('000[0-9]+', path_components[-1])[-1]
        dataset_name = path_components[-3]

        os.makedirs(os.path.join(args.model_dir, dataset_name), exist_ok=True)
        pkl_file_path = os.path.join(args.model_dir, dataset_name, model_id + ".pkl")
        with open(pkl_file_path, "wb") as f:
            pickle.dump(activations, f)
        pkl_file_paths.append(pkl_file_path)

    model_ref_acc = model_ref_corr / args.samples

    print(f"\nRan through {args.samples} samples from {len(set(image_labels[:args.samples]))} categories. \nAccuracy: ref {model_ref_acc}.")

    return pkl_file_paths


def run_model(args: argparse.Namespace, image_paths: str, image_labels: str, ref_labels: str):
    # Force PyTorch to CPU
    torch.device("cpu")

    # Create reference model and fill with weights
    model_ref = resnet_reference.ResNet(
        block=resnet_reference.BasicBlock, layers=[2, 2, 2, 2])
    # ref_state_dict = load_state_dict_from_url(
    #     "https://download.pytorch.org/models/resnet18-5c106cde.pth")
    # model_ref.load_state_dict(ref_state_dict)

    # Create bn folded + maxpool 2x2 model (bn folded into conv)
    model_bnfold = vision_models.ResNet(
        block=vision_models.BasicBlock, layers=[2, 2, 2, 2])
    model_path = os.path.join(args.model_dir, 'resnet18_mp2_p8_qat.pth')
    bnfold_state_dict = torch.load(model_path, map_location=torch.device('cpu'))['state_dict']
    model_bnfold.load_state_dict(bnfold_state_dict)
    model_bnfold.eval()
    model_bnfold = bn_folding.bn_folding_model(model_bnfold)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if args.write_model:
        output_folder = os.path.join(args.output_dataset_dir, "params")
        os.makedirs(output_folder, exist_ok=True)

        for name, param in model_bnfold.named_parameters():
            data = arrange_data(name, param.data)
            file_name = re.sub('downsample.0', 'downsample', name)
            file_name = re.sub('\.', '_', file_name)

            with open(os.path.join(output_folder, file_name), 'wb') as output:
                output.write(struct.pack('%sd' % len(data), *data))

    model_ref_corr = 0
    model_bnfold_corr = 0
    pkl_file_paths = []

    vision_models._no_intermediates = args.no_intermediates
    vision_models._no_weights = args.no_weights

    with open(os.path.join(args.data_dir, 'labels.txt'), 'r') as f:
        class_labels = f.readlines()
        class_labels = [s.split(',')[0] for s in class_labels]

    # Loop over all images and verify results
    for i in tqdm(range(args.samples)):

        image_path = image_paths[i]

        # Load and prepare input image
        input_image = Image.open(image_path).convert('RGB')
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

        # TODO(fpedd): Models have a different architecture, so they won't be bit accurate
        # # Compare output between models
        # diff = functools.reduce(
        #     (lambda acc, top: acc or top[0] != top[1]), zip(top_catid_ref, top_catid_bnfold), False)
        # if diff:
        #     raise ValueError("Top 5 differ at {} with {} vs {}".format(
        #         image_labels[i], top_catid_ref, top_catid_bnfold))

        if image_labels[i] == ref_labels[top_catid_ref[0]][0]:
            model_ref_corr += 1

        if image_labels[i] == ref_labels[top_catid_bnfold[0]][0]:
            model_bnfold_corr += 1

        if vision_models._export:

            # Get unique name from image path
            path_components = os.path.normpath(image_path).split(os.path.sep)
            model_id = path_components[-2] + '_' + \
                re.findall('000[0-9]+', path_components[-1])[-1]
            dataset_name = path_components[-3]

            os.makedirs(os.path.join(args.model_dir,
                        dataset_name), exist_ok=True)
            pkl_file_path = os.path.join(
                args.model_dir, dataset_name, model_id + ".pkl")
            with open(pkl_file_path, "wb") as f:
                pickle.dump(vision_models._buffer, f)
            pkl_file_paths.append(pkl_file_path)

            # Write images to disk
            if args.write_dataset:
                imagenet_label = ref_labels[top_catid_ref[0]][0]
                class_id = class_labels.index(imagenet_label)
                sample_name = f"{class_id}_{model_id}"
                output_folder_name = os.path.join(args.output_dataset_dir,sample_name)
                os.makedirs(output_folder_name, exist_ok=True)
                
                input_data = vision_models._buffer['conv1.input']
                # Open output file, then pack and write data

                with open(os.path.join(output_folder_name, "input"), 'wb') as output:
                    output.write(struct.pack('%sd' % len(input_data), *input_data))

            if args.export_onnx:
                # Write model to disk
                onnx_model_file_path = os.path.join(
                    args.model_dir, "resnet18" + ".onnx")
                torch.onnx.export(model_bnfold, input,
                                  onnx_model_file_path, verbose=False)
                # Load model again using ONNX
                model_bnfold_onnx = onnx.load(onnx_model_file_path)
                # Check that the model is well formed
                onnx.checker.check_model(model_bnfold_onnx)
                # Do shape inference on ONNX and save
                model_bnfold_onnx_inferred = onnx.shape_inference.infer_shapes(
                    model_bnfold_onnx)
                onnx_model_file_path_inferred = os.path.join(
                    args.model_dir, "resnet18_inferred" + ".onnx")
                onnx.save(model_bnfold_onnx_inferred,
                          onnx_model_file_path_inferred)
                # TODO(fpedd): We only want to save one onnx model currently
                args.export_onnx = False

            vision_models.reset()

    model_ref_acc = model_ref_corr / args.samples
    model_bnfold_acc = model_bnfold_corr / args.samples

    print(
        f"\nRan through {args.samples} samples from {len(set(image_labels[:args.samples]))} categories. \nAccuracy: ref {model_ref_acc} bnfold {model_bnfold_acc}.")

    return pkl_file_paths


def write_binary(args: argparse.Namespace, pkl_file_paths):
    print("Writing binary files to: {}".format(args.binary_dir))

    # Create output_dir if it does not already exists
    os.makedirs(args.binary_dir, exist_ok=True)

    for pkl_file_path in tqdm(pkl_file_paths):

        instance = os.path.splitext(os.path.basename(pkl_file_path))[0]
        output_dir = os.path.join(args.binary_dir, instance)
        os.makedirs(output_dir, exist_ok=True)

        # Load pickle file and write binary layer files
        with open(pkl_file_path, 'rb') as f:

            # Working with posit8 datatypes
            if args.data_type == 'posit8':

                # Iterate over all entries in the pickled dictionary
                for name, data in pickle.load(f).items():

                    # Convert name to filename (special case for downsample)
                    file_name = re.sub('downsample.0', 'downsample', name)
                    file_name = re.sub('\.', '_', file_name)

                    # Open output file, then pack and write data
                    with open(os.path.join(output_dir, file_name), 'wb') as output:
                        output.write(struct.pack('%sd' % len(data), *data))
            else:
                raise ValueError(
                    "Unsupported datatype {}!".format(args.data_type))


def dump_model_weights(args):
    model = models.__dict__['resnet18']()
    model.eval()
    model = bn_folding.bn_folding_model(model)

    ckpt_file = os.path.join(args.model_dir, args.model_file)
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = arrange_data(name, param.data)

    pkl_file = os.path.join(args.binary_dir, "weights.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(weights, f)

    return [pkl_file]


def main():

    parser = argparse.ArgumentParser(
        description='ResNet data generator (from model and images to binary data).')
    parser.add_argument('--data_dir',
                        type=str,
                        default='data/imagenet',
                        help='Path to dataset.')
    parser.add_argument('--model_dir',
                        type=str,
                        default='models/resnet/model_data',
                        help='Path to model directory the .pkl intermediate files will be written to.')
    parser.add_argument('--model_file',
                        type=str,
                        default='model_best.pth.tar',
                        help='Path to model file the .pkl intermediate files will be written to.')
    parser.add_argument('--binary_dir',
                        type=str,
                        default='models/resnet/binary_data',
                        help='Path the binary data output will be written to (binary files for simulation).')
    parser.add_argument('--write_dataset', default=False, action='store_true', help='Write dataset to disk.')
    parser.add_argument('--write_model', default=False, action='store_true', help='Write model weights to disk.')
    parser.add_argument('--output_dataset_dir', type=str, default='data/imagenet', help='Path to output folder for dataset.')
    parser.add_argument('--samples',
                        type=int,
                        default=1,
                        help='How many samples to generate.')
    parser.add_argument('--no_intermediates',
                        default=False,
                        action='store_true',
                        help='Generate no intermediate buffers, only entry and exit.')
    parser.add_argument('--no_weights',
                        default=False,
                        action='store_true',
                        help='Generate no weight data.')
    parser.add_argument('--data_type',
                        type=str,
                        default='posit8', help='Datatype to parse (posit8, int8).')
    parser.add_argument('--export_onnx',
                        default=False,
                        action='store_true',
                        help='Export bn folded .onnx model.')
    args = parser.parse_args()

    image_paths, image_labels, ref_labels = prepare_data(args)
    assert len(image_paths) >= args.samples, f'Can not generate {args.samples}, only {len(image_paths)} samples available.'
    # pkl_file_paths = run_model(args, image_paths, image_labels, ref_labels)
    pkl_file_paths = dump_model_data(args, image_paths, image_labels, ref_labels)
    # pkl_file_paths = dump_model_weights(args)
    write_binary(args, pkl_file_paths)

    print("ResNet data generator done!")


if __name__ == "__main__":
    main()
