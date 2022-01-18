import dill as pickle
import numpy as np
import torch
import re

def get_layer_size(arr):
    result = 1
    dims = np.shape(arr)
    for num in dims:
        result *= num
    return result

def itoba(num):
    arr = []
    for i in range(4):
        arr.insert(0, num & 0b11111111)
        num = num >> 8;

    arr = reversed(arr)
    return arr

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

# folder = 'layer_data'
infile = open("resnet8.pkl",'rb')
new_dict = pickle.load(infile)

for key, value in new_dict.items():
    # print(key)
    # print(get_layer_size(value))

    filename = "resnet/" + clean_name(key)
    print(filename)
    with open(filename, 'wb') as f:
        # f.write(bytearray(itoba(get_layer_size(value))))
        byte_list = []
        
        for i in np.nditer(value):
            if (i > 256):
                print(i)
            if (i < 0):
                byte_list.append(i + 256)
            else:
                byte_list.append(i)

        f.write(bytearray(byte_list))

        

        # f.write(bytearray([100, 100]))

infile.close()
