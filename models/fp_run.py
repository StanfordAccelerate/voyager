import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy
import dill as pickle
import numpy as np
import copy
import fp_models
import bn_folding

# Stupid Imagenet conversion map
lines = []                                                                                                                         
with open('/sim/kbartolo/accelerator/data/imagenet/clscmap.txt', 'r') as f: 
    lines = f.readlines()

# Split out and drop empty rows
strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']

clscmap = {}
i = 1
for strip in strip_list: 
    clscmap[strip[0]] = i
    i +=1

# Disable cuda for char tensors
torch.backends.cudnn.enabled = True

# Load base resnet18
gold_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
gold_model.eval()

# Load bn folded comparison model
comp_model = fp_models.resnet18()
comp_model.load_state_dict(torch.load("/sim/kbartolo/accelerator/models/resnet18_mp2.pth").state_dict(), strict=False)
# comp_model.load_state_dict(gold_model.state_dict())
comp_model.eval()
comp_model = bn_folding.bn_folding_model(comp_model)

# Evaluation header
with open("../data/imagenet/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
with open("../data/imagenet/ILSVRC2012_validation_ground_truth.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

sample_size = 1
gold_correct = 0.0
comp_correct = 0.0

index = 0
samples = 0
while(samples < sample_size):
    image_index = index + 1
    print(index)

    filename = "/sim/kbartolo/accelerator/data/imagenet/ILSVRC2012_val_"+"{:08d}".format(image_index)+".jpeg"
    filename = "/sim/kbartolo/accelerator/data/imagenet/ILSVRC2012_val_"+"{:08d}".format(0)+".jpeg"

    # Create Mini-Batch
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        input_tensor = preprocess(input_image)
    except RuntimeError:
        index+=1
        continue
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Run batch gold
    # print("Gold")
    with torch.no_grad():
        output = gold_model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 1)
    # print(clscmap[categories[top5_catid[0]]])
    # print(labels[index])

    if (clscmap[categories[top5_catid[0]]] == int(labels[index])):
        gold_correct += 1
        # print(gold_correct, sample_size)

    with torch.no_grad():
        output = comp_model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 1)
    if (clscmap[categories[top5_catid[0]]] == int(labels[index])):
        comp_correct += 1

    # fp_models.complete["label"] = int(labels[index])
    # file = open('accuracy/'+"test_{:08d}".format(samples)+'.pkl', 'wb')
    # pickle.dump(fp_models.complete, file)
    file = open('resnet_pretrained.pkl', 'wb')
    pickle.dump(fp_models.complete, file)
    index += 1    
    samples += 1
    print("comp: " + str(comp_correct/samples) + " vs gold: " + str(gold_correct/samples))

final = "comp: 0.695 vs gold: 0.702"
