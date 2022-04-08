#!/usr/bin/python3
import dill as pickle
import numpy as np
import torch
import re
import sys, getopt
import os
import struct


''' @brief: Writes data of form torch.tensor dtype=float64 to binary data. '''
def write_fp64(filename, data):
    # data = data.type(torch.float64)
    data = data.astype(np.float64)
    with open(filename, 'wb') as f:
        floatlist = []
        
        for i in np.nditer(data):
            floatlist.append(i)

        buf = struct.pack('%sd' % len(floatlist), *floatlist)
        f.write(buf)

''' @brief: Returns size of flattened np array. ''' 
def get_layer_size(arr):
    result = 1
    dims = np.shape(arr)
    for num in dims:
        result *= num
    return result

''' @brief: Returns number as byte array. '''
def itoba(num):
    arr = []
    for i in range(4):
        arr.insert(0, num & 0b11111111)
        num = num >> 8

    arr = reversed(arr)
    return arr

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

if __name__ == "__main__":
    # Get input file and output folder
    argv = sys.argv[1:]
    inputfile = ''
    outputfolder = ''
    datatype = ''
    try:
       opts, args = getopt.getopt(argv,"hi:o:t:",["ifile=","ofile=", "type="])
    except getopt.GetoptError:
       print('usage: pkl_parser.py -i <inputfile> -o <outputfolder> -t <type>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('usage: pkl_parser.py -i <inputfile> -o <outputfolder> -t <type>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputfolder = arg
       elif opt in ("-t", "--type"):
          datatype = arg

    # Create a new directory because it does not exist 
    folder_exists = os.path.exists(os.getcwd() + "/" + outputfolder)
    if not folder_exists:
        os.makedirs(os.getcwd() + "/" + outputfolder)

    # Load pickle file and write binary layer files
    infile = open(inputfile,'rb')
    new_dict = pickle.load(infile)

    if (datatype == 'int8'):
        for key, value in new_dict.items():
            filename = outputfolder + "/" + clean_name(key)
            print(filename)
            with open(filename, 'wb') as f:
                byte_list = []
        
                for i in np.nditer(value):
                    if (i < 0):
                        byte_list.append(i + 256)
                    else:
                        byte_list.append(i)

                f.write(bytearray(byte_list))
    elif (datatype == 'posit8'):
        for key, value in new_dict.items():
            prefilename = outputfolder + "/" + "pre" + clean_name(key)
            postfilename = outputfolder + "/" + clean_name(key)
            write_fp64(postfilename, value)
            #os.system("./tools/decode " + prefilename + " " + postfilename)
            print(postfilename)

    infile.close()
