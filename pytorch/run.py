import torch
import torch.nn as nn
from torchvision import transforms
import resnet18_fp16
from PIL import Image
import numpy
import dill as pickle
import numpy as np
import copy
import resnet_gen
import bn_folding

# Disable cuda for char tensors
torch.backends.cudnn.enabled = True

# Load modified model
gold_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
gold_model.eval()
gold_model = gold_model.type(torch.cuda.HalfTensor)

bn_model = bn_folding.bn_folding_model(gold_model)
bn_model.eval()

comp_model = resnet_gen.resnet18().type(torch.cuda.HalfTensor)
comp_model.eval()
comp_model = bn_folding.bn_folding_model(comp_model)
comp_model.load_state_dict(bn_model.state_dict())

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "data/dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# Create Mini-Batch
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.type(torch.cuda.HalfTensor)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# Run batch gold
with torch.no_grad():
    output = gold_model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Download ImageNet labels

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# Run batch bn
with torch.no_grad():
    output = bn_model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Download ImageNet labels

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

# Run batch comp
with torch.no_grad():
    output = comp_model(input_batch)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Download ImageNet labels

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())