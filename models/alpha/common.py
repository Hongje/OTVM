from __future__ import division
import torch.nn as nn
import torch.nn.functional as F


def pad_divide_by(in_list, d, in_size, padval=0.):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    if sum(pad_array)>0:
        for inp in in_list:
            out_list.append(F.pad(inp, pad_array, value=padval))
    else:
        out_list = in_list
    if len(in_list) == 1:
        out_list = out_list[0]
    return out_list, pad_array
