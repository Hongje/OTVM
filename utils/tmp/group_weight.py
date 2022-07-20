import torch
import torch.nn as nn
from models.models_TCVOM.FBA import layers_WS

def group_weight(module, lr_encoder, lr_decoder, WD):
    group_decay = { 'encoder': [], 'decoder':[]}
    group_bias = { 'encoder': [], 'decoder':[]}
    group_GN = { 'encoder': [], 'decoder':[]}


    for name, m in module.named_modules():
        # if hasattr(m, 'requires_grad'):
        #     if m.requires_grad:
        #         continue
        
        part = 'decoder'
        if('encoder' in name):
            part = 'encoder'

        if isinstance(m, nn.Linear):
            group_decay[part].append(m.weight)
            if m.bias is not None:
                group_bias[part].append(m.bias)

        elif isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            group_decay[part].append(m.weight)
            if m.bias is not None:
                group_bias[part].append(m.bias)
        elif isinstance(m, layers_WS.Conv2d) and m.weight.requires_grad:
            group_decay[part].append(m.weight)
            if m.bias is not None:
                group_bias[part].append(m.bias)

        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_GN[part].append(m.weight)
            if m.bias is not None:
                group_GN[part].append(m.bias)
        

    print(len(list(module.parameters())), len(group_decay['encoder']) + len(group_bias['encoder']) + len(group_GN['encoder']) + len(group_decay['decoder']) + len(group_bias['decoder']) + len(group_GN['decoder']) , len(list(module.modules())))
    # assert len(list(module.parameters())) == len(group_decay) + len(group_bias) + len(group_GN)
    groups = [dict(params=group_decay['decoder'], lr =lr_decoder, weight_decay=WD), dict(params=group_bias['decoder'], lr=2*lr_decoder, weight_decay=0.0), dict(params=group_GN['decoder'], lr=lr_decoder, weight_decay=1e-5),
    dict(params=group_decay['encoder'], lr=lr_encoder, weight_decay=WD), dict(params=group_bias['encoder'], lr=2*lr_encoder, weight_decay=0.0), dict(params=group_GN['encoder'], lr=lr_encoder, weight_decay=1e-5)]

    # groups= [dict(params=module.decoder.conv_pred.parameters(), lr=lr, weight_decay=0.0)]
    return groups