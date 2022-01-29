import torch
import torch.nn as nn
import numpy as np
import copy

### brief: Returns bn folded model
def bn_folding_model(model):

    new_model = copy.deepcopy(model)

    module_names = list(new_model._modules)

    for k, name in enumerate(module_names):
        
        if len(list(new_model._modules[name]._modules)) > 0:
            new_model._modules[name] = bn_folding_model(new_model._modules[name])
            
        else:
            if isinstance(new_model._modules[name], nn.BatchNorm2d):
                if isinstance(new_model._modules[module_names[k-1]], nn.Conv2d):

                    # Folded BN
                    folded_conv = fold_conv_bn_eval(new_model._modules[module_names[k-1]], new_model._modules[name])

                    # Replace old weight values
                    new_model._modules[(name)] = torch.nn.Identity() # Remove the BN layer
                    new_model._modules[module_names[k-1]] = folded_conv # Replace the Convolutional Layer by the folded version
        
    return new_model

def bn_folding(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = bn_rm.new_zeros(bn_rm.shape)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    
    w_fold = conv_w * (bn_w * bn_var_rsqrt).view(-1, 1, 1, 1)
    b_fold = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    
    return torch.nn.Parameter(w_fold), torch.nn.Parameter(b_fold)


def fold_conv_bn_eval(conv, bn):
    assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = bn_folding(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv

def get_folded_weights(conv_weights, batch_norm_weights, batch_norm_bias, batch_norm_mean, batch_norm_var, epsilon = 1e-3):
  gamma = batch_norm_weights.reshape((1, 1, 1, batch_norm_weights.shape[0]))
  beta = batch_norm_bias
  mean = batch_norm_mean
  variance = batch_norm_var.reshape((1, 1, 1, batch_norm_var.shape[0]))

  new_weights = (conv_weights[0] * gamma / np.sqrt(variance + epsilon))
  new_bias = beta + (conv_weights[1] - mean) * gamma / np.sqrt(variance + epsilon)
  return new_weights, new_bias