from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from helpers import ToCuda, pad_divide_by
import math

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self, hdim=32):
        super(Encoder_M, self).__init__()
        self.hdim = hdim

        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.hdim > 0:
            self.conv1_a = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_h = nn.Conv2d(hdim, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o, in_a, in_h):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim
        if self.hdim > 0:
            a = torch.unsqueeze(in_a, dim=1).float() # add channel dim
            h = in_h.float() # add channel dim
            x = self.conv1_m(m) + self.conv1_o(o) + self.conv1_a(a) + self.conv1_h(h)
        else:
            x = self.conv1_m(m) + self.conv1_o(o)

        x = self.conv1(f) + x
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred = nn.Conv2d(mdim, 3, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, VOS_mode=False):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T*H*W) 
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
 
        p = torch.bmm(mi, qi) # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1) # b, THW, HW

        mo = m_out.view(B, D_o, T*H*W) 
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out


class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)




class STM(nn.Module):
    def __init__(self, hdim=-1):
        super(STM, self).__init__()
        self.hdim = hdim

        self.Encoder_M = Encoder_M(hdim=self.hdim)
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
 
    def Pad_memory(self, mems, num_objects):
        pad_mems = []
        for mem in mems:
            batch_and_numobj, C, H, W = mem.shape
            batch_size = batch_and_numobj//num_objects
            pad_mems.append(mem.view(num_objects, batch_size, C, 1, H, W).transpose(1,0))
        return pad_mems

    def memorize(self, frame, masks, num_objects): 
        # memorize a frame 
        num_objects = num_objects[0].item()

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[], 'a':[], 'h':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,1]) # Unkown region
            B_list['o'].append(masks[:,2]) # Foreground region
            if self.hdim > 0:
                B_list['a'].append(masks[:,3]) # Alpha matte
                B_list['h'].append(masks[:,4:]) # hidden layer

        # make Batch
        B_ = {}
        B_['a'] = None
        B_['h'] = None
        for arg in B_list.keys():
            if len(B_list[arg]) > 0:
                B_[arg] = torch.cat(B_list[arg], dim=0)

        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'], B_['a'], B_['h'])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects)
        return k4, v4

    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit

    def segment(self, frame, keys, values, num_objects):
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        
        # memory select kv:(1, K, C, T, H, W)
        m4 = self.Memory(keys.squeeze(1), values.squeeze(1), k4, v4)
        logits = self.Decoder(m4, r3, r2)
        
        logit = logits

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit    

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)

