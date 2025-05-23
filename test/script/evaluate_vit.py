import argparse
import os
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTForImageClassification


def get_transforms(model_type):
    if model_type == "resnet":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif model_type == "vit":
        return transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_vit_on_local_imagenet(local_imagenet_root_path, num_samples=None):
    print(
        f"Starting evaluation on local ImageNet data from: {local_imagenet_root_path}"
    )

    vit_transforms = get_transforms("vit")

    try:
        full_dataset = datasets.ImageFolder(
            local_imagenet_root_path, transform=vit_transforms
        )
        print(f"Found {len(full_dataset)} samples in {local_imagenet_root_path}")
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    dataset_to_evaluate = full_dataset
    if num_samples is not None:
        dataset_to_evaluate = torch.utils.data.Subset(
            full_dataset, range(min(len(full_dataset), num_samples))
        )

    imagenet_n_id_to_name = {}
    labels_file_path = os.path.join(local_imagenet_root_path, "labels.txt")
    if not os.path.exists(labels_file_path):
        print(f"Error: labels.txt not found at {labels_file_path}.")
        return None

    try:
        with open(labels_file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    n_id = parts[0]
                    primary_class_name = parts[1].split(",")[0].strip()
                    imagenet_n_id_to_name[n_id] = primary_class_name
    except Exception as e:
        print(f"Error reading labels.txt: {e}")
        return None

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("ImageFolder class to index mapping:", full_dataset.class_to_idx)
    # print("Model label to index mapping:", model.config.label2id)
    print("n_id to human-readable name mapping:", imagenet_n_id_to_name)

    imagenet_folder_idx_to_model_idx = {}
    num_unmapped_classes = 0
    for folder_name, folder_idx in full_dataset.class_to_idx.items():
        human_readable_name = imagenet_n_id_to_name.get(folder_name)
        if human_readable_name:
            model_expected_idx = model.config.label2id.get(human_readable_name)
            if model_expected_idx is not None:
                imagenet_folder_idx_to_model_idx[folder_idx] = model_expected_idx
            else:
                print(
                    f"Warning: Model does not have a mapping for '{human_readable_name}' (from n-ID: {folder_name})."
                )
                num_unmapped_classes += 1
        else:
            print(f"Warning: n-ID '{folder_name}' not found in labels.txt.")
            num_unmapped_classes += 1

    if num_unmapped_classes > 0:
        print(f"Note: {num_unmapped_classes} classes could not be fully mapped.")
    if not imagenet_folder_idx_to_model_idx:
        print("Error: No classes could be successfully mapped.")
        return None

    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset_to_evaluate, batch_size=batch_size, shuffle=False, num_workers=4
    )

    correct_predictions = 0
    total_samples = 0

    print("Starting evaluation loop...")
    with torch.no_grad():
        for images, labels_from_imagefolder in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            remapped_labels = []
            valid_batch_indices = []
            original_indices = []

            for i, label in enumerate(labels_from_imagefolder):
                original_index = label.item()
                model_target_index = imagenet_folder_idx_to_model_idx.get(
                    original_index
                )
                if model_target_index is not None:
                    remapped_labels.append(model_target_index)
                    valid_batch_indices.append(i)
                    original_indices.append(original_index)
                else:
                    print(
                        f"Warning: Skipping sample with ImageFolder index {original_index} as no model mapping found."
                    )

            if not valid_batch_indices:
                continue  # Skip the batch if no valid mappings

            remapped_labels_tensor = torch.tensor(remapped_labels, device=device)
            valid_images = images[valid_batch_indices]

            outputs = model(valid_images)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            correct_predictions += (predictions == remapped_labels_tensor).sum().item()
            total_samples += remapped_labels_tensor.size(0)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Accuracy on the local ImageNet validation subset: {accuracy:.4f}")
    return accuracy


def main():
    torch.set_num_threads(32)
    parser = argparse.ArgumentParser(description="Evaluate ViT on local ImageNet.")
    parser.add_argument(
        "--data_dir", default="data/imagenet", help="Path to ImageNet root."
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate."
    )
    args = parser.parse_args()
    accuracy = evaluate_vit_on_local_imagenet(args.data_dir, args.num_samples)
    if accuracy is not None:
        print(f"Final Evaluated Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
