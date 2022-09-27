import sys, getopt
sys.path.insert(1, '/sim/kbartolo/accelerator/models')
import dill as pickle
import numpy as np
import struct
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import fp_models
import bn_folding
import subprocess
from subprocess import PIPE, STDOUT

''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    # data = data.type(torch.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)

def write_fp64fd(f, data):
    # data = data.type(torch.float64)
    floatlist = []
       
    for i in np.nditer(data):
        floatlist.append(i)

    buf = struct.pack('%sd' % len(floatlist), *floatlist)
    f.write(buf)

''' @brief: Changes . to _ and fixes downsample layer names. '''
def clean_name(name):
    result = ""
    # if (not re.fullmatch("(.*bn2.*)|(.*bn1.*)|(.*downsample.*)", name)):
    if (name.find("downsample.0") != -1):
        name = name[0:name.find("downsample.0")] + "downsample" + name[name.find("downsample.0") + len("downsample.0"):]
    for c in name:
        if (c == '.'):
            result += '_'
        else:
            result += c
    
    return result

buildfolder = "pybuild"
imagenetfolder = "data/imagenet"
toolfolder = "tools"

if __name__ == "__main__":
    # Get input file and output folder
    argv = sys.argv[1:]
    start_index = 0
    end_index = 0
    id = 0
    try:
       opts, args = getopt.getopt(argv,"hs:e:d:",["start=","end=", "id="])
    except getopt.GetoptError:
       print('usage: evaluate_accuracy_sub.py -s <start_index> -e <end_index>, -d <id>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('usage: evaluate_accuracy_sub.py -s <start_index> -e <end_index>, -d <id>')
          sys.exit()
       elif opt in ("-s", "--start"):
          start_index = int(arg)
       elif opt in ("-e", "--end"):
          end_index = int(arg)
       elif opt in ("-d", "--id"):
          id = int(arg)

    # Open validation translation files
    with open(imagenetfolder+"/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    with open(imagenetfolder+"/ILSVRC2012_validation_ground_truth.txt", "r") as f:
        labels = [s.strip() for s in f.readlines()]

    # Stupid Imagenet conversion map
    lines = []                                                                                                                         
    with open(imagenetfolder+'/clscmap.txt', 'r') as f: 
        lines = f.readlines()

    # Split out and drop empty rows
    strip_list = [line.replace('\n','').split(' ') for line in lines if line != '\n']

    # Read number to path conversion map
    imgtopath = []
    with open(os.path.join(imagenetfolder, "imgtopath.txt"), 'r') as f:
        imgtopath = f.readlines()
    imgtopath = [line.replace('\n', '') for line in imgtopath]

    clscmap = {}
    i = 1 
    for strip in strip_list: 
        clscmap[strip[0]] = i
        i +=1

    # Load base resnet18
    gold_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    gold_model.eval()

    # Load bn folded comparison model
    comp_model = fp_models.resnet18()
    comp_model.load_state_dict(torch.load("/sim/kbartolo/accelerator/models/resnet18_mp2.pth").state_dict(), strict=False)
    comp_model.eval()
    comp_model = bn_folding.bn_folding_model(comp_model)


    # Evaluate accuracy for slice
    samples = 0.0
    sim_correct = 0.0
    gold_correct = 0.0
    comp_correct = 0.0

    # Writing files
    blacklist_file = open(os.path.join(buildfolder, "blacklist" + str(id)), 'w')
    acc_file = open(os.path.join(buildfolder, "accuracy" + str(id)), 'w')

    for sample in range(start_index, end_index):
        # Grab input
        filename = imgtopath[sample - 1]

        # Perform transformation and upon failure blacklist image
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
            blacklist_file.write(filename + '\n')
            continue
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        
        os.environ["MODEL"] = "resnet"
        os.environ["TEST"] = "conv1"
        p = subprocess.Popen(["./build/TestRunner"], stdin=PIPE, stdout=PIPE)
        # p = subprocess.Popen(["wc"], stdin=PIPE, stdout=PIPE)
        write_fp64fd(p.stdin, fp_models.arrange_data("input", input_batch))
        # p.stdin.write(b"index12u")

        with torch.no_grad():
            output = gold_model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 1)

        if (clscmap[categories[top5_catid[0]]] == int(labels[sample - 1])):
            gold_correct += 1

        with torch.no_grad():
            output = comp_model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 1)
        if (clscmap[categories[top5_catid[0]]] == int(labels[sample - 1])):
            comp_correct += 1
        streamdata = str(p.communicate()[0])
        start = streamdata.find("index") + len("index")
        end = streamdata.find("u", start)
        index = int(streamdata[start:end])
        p.wait()
        if (clscmap[categories[index]] == int(labels[sample - 1])):
            sim_correct += 1
 

        samples += 1
        acc_statement = "sim: " + str(sim_correct/samples) + ", comp: " + str(comp_correct/samples) + ",gold: " + str(gold_correct/samples) + '\n'

        acc_file.write(acc_statement)
        acc_file.flush()
    acc_file.close()
    blacklist_file.close()

''' Pseudo-code
for sample in range(start_sample, end_sample):
    grab file
    if not valid
        update blacklist
    else 
        cat data/resnet/conv1_input | MODEL=resnet TEST=conv1 ./build/TestRunner
        process = subprocess.Popen(['foo', '-b', 'bar'])
        perform run
        update accuracy
    



'''

