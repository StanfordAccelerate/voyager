# ResNet Model
This directory contains the PyTorch ResNet model. Use `gen_resnet_data.py` to generate ResNet model data (weights and activations). You need to generate the binary before you can run the SystemC ResNet accelerator simulations.

## Function
- `gen_resnet_data.py` runs the model in `resnet_record.py`, records the weight and activation tensors, and stores them in a `.pkl` file in the `model_data/` directory. The `.pkl` has no immediate use and is only used as an intermediate storage form.
- In a next step `gen_resnet_data.py` takes the `.pkl` and generates binary files that are used by the simulator. The binary files are written to `binary_data/`.

## Info
- `model_data/` contains the intermediate `.pkl` files, as well as a model `state_dict`s (`.pth` ot `.pt`) which can be used as an alternative to downloading models from the internet.
- The tensors (activations/weights) in the `.pkl` files are stored in a Python dict; key: layer name, value: weight/activation tensor.
- `binary_data/` will contain the binary data used by the simulator
- The model in `resnet_record.py` is a modified version of the "reference" model in `resnet_reference.py`, which in turn is a slightly modified and striped-down version of the "official" [PyTorch model](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) (copied at commit a89b195 (22/09/2022)). 
- `bn_folding.py` contains functions to fold the batch norm layers of PyTorch models into the conv layers.