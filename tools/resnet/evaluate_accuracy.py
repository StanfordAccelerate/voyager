import dill as pickle
import numpy as np
import struct
import os

''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    # data = data.type(torch.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            # print(i)
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

# Test settings
sample_size = 1000
buildfolder = "pybuild"
imagenetfolder = "data/imagenet"
toolfolder = "tools"

# Creat build folder if it does not yet exist
folder_exists = os.path.exists(os.getcwd() + "/" + buildfolder)
if not folder_exists:
    os.makedirs(os.getcwd() + "/" + buildfolder)

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

clscmap = {}
i = 1
for strip in strip_list: 
    clscmap[strip[0]] = i
    i +=1

samples = 0
correct_samples = 0.0
while(samples < sample_size):
    # Grab input
    filename ="/sim/kbartolo/accuracy/test_{:08d}".format(samples)+".pkl"
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    print(filename, new_dict["label"])
    for key, value in new_dict.items():
        if key != "conv1.input":
            continue
        prefilename = buildfolder + "/" + "pre" + clean_name(key)
        #MOD
        postfilename = "data/resnet" + "/" + clean_name(key)

        # Convert input to posit
        # write_fp64(prefilename, value)
        os.system("rm data/resnet/conv1_input")

        write_fp64(postfilename, value)
        # os.system(toolfolder+"/decode " + prefilename + " " + postfilename)
        print(postfilename)
    infile.close()

    # Run complete simulation to file
    os.system("make -j sim MODEL=resnet TEST=conv1")


    # Convert output to floating point and get max
    # os.system(toolfolder + "/interpreter "+buildfolder+"/output "+buildfolder+"/result.txt")
    
    with open(buildfolder + "/result.txt", "r") as f:
        result = [s.strip() for s in f.readlines()]

    samples += 1

    # Compare max to label list and check if right or wrong
    # if (int(result[0]) == new_dict("label")
    if (int(clscmap[categories[int(result[0])]]) == new_dict["label"]):
        correct_samples += 1
        
    # Print running accuracy
    print("sim caffe: "+result[0]+ ", sim: " + str(clscmap[categories[int(result[0])]])+" vs correct: " + str(new_dict["label"]))
    print("running accuracy: " + str(correct_samples / samples))

    with open("accuracy.txt", 'a') as f:
        acc_msg = "sim caffe: "+result[0]+ ", sim: " + str(clscmap[categories[int(result[0])]])+" vs correct: " + str(new_dict["label"]) +"\nrunning accuracy: " + str(correct_samples / samples) +"\n"
        f.write(acc_msg)
