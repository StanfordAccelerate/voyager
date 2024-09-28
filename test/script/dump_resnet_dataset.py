import argparse
import os
import torch

from torchvision import datasets, transforms
from tqdm import tqdm

from utils import write_tensor_to_file


def dump_imagenet(data_dir, output_dir, num_samples):
    dataset = datasets.ImageFolder(
        data_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))

    if num_samples is None:
        num_samples = len(dataset)

    assert len(dataset) >= num_samples, (
        f'Can not generate {num_samples}, only {len(dataset)} samples available.'
    )

    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        class_labels = f.readlines()
        class_labels = [s.split(',')[0] for s in class_labels]


    # Iterate over the selected indices and retrieve samples
    for i in tqdm(range(num_samples)):
        image, _ = dataset[i]
        path, _ = dataset.samples[i]

        # Extract the directory name (class ID)
        class_id = os.path.basename(os.path.dirname(path))
        label = class_labels.index(class_id)

        # Extract the file name and split to get the image identifier
        file_name = os.path.basename(path)
        image_id = file_name.split('_')[-1].split('.')[0]

        dir_name = os.path.join(output_dir, f"{label}_{class_id}_{image_id}")
        os.makedirs(dir_name, exist_ok=True)
        write_tensor_to_file(image, os.path.join(dir_name, "arg0_1.bin"))


def main():
    torch.set_num_threads(32)

    parser = argparse.ArgumentParser(
        description='ResNet data generator (from model and images to binary data).'
    )
    parser.add_argument(
        '--data_dir',
        default='data/imagenet',
        help='Path to dataset.',
    )
    parser.add_argument(
        '--output_dir',
        default='data/imagenet',
        help='Path to output folder for dataset.',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='How many samples to generate.',
    )
    args = parser.parse_args()
    dump_imagenet(args.data_dir, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
