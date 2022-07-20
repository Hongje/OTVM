from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
 
# general libs
import sys

sys.path.insert(0, '../')
from helpers import *

from .STM import STM


class FullModel(nn.Module):
    def __init__(self, dilate_kernel=None, eps=0, ignore_label=255,
                 stage=1,
                 hdim=-1,):
        super(FullModel, self).__init__()
        self.DILATION_KERNEL = dilate_kernel
        self.EPS = eps
        self.IMG_SCALE = 1./255
        self.register_buffer('IMG_MEAN', torch.tensor([0.485, 0.456, 0.406]).reshape([1, 1, 3, 1, 1]).float())
        self.register_buffer('IMG_STD', torch.tensor([0.229, 0.224, 0.225]).reshape([1, 1, 3, 1, 1]).float())
        
        self.stage = stage
        self.hdim = hdim if self.stage > 2 else -1
        self.memory_update = False

        self.model = STM(hdim=self.hdim)
        
        self.num_object = 1

        self.ignore_label = ignore_label
        self.LOSS = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1]).float(), ignore_index=ignore_label)

    def make_trimap(self, alpha, ignore_region):
        b = alpha.shape[0]
        alpha = torch.where(alpha < self.EPS, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha > 1 - self.EPS, torch.ones_like(alpha), alpha)
        trimasks = ((alpha > 0) & (alpha < 1.)).float().split(1)
        trimaps = [None] * b
        for i in range(b):
            # trimap width: 1 - 51
            kernel_rad = int(torch.randint(0, 26, size=())) \
                if self.DILATION_KERNEL is None else self.DILATION_KERNEL
            trimaps[i] = F.max_pool2d(trimasks[i].squeeze(0), kernel_size=kernel_rad*2+1, stride=1, padding=kernel_rad)
        trimap = torch.stack(trimaps)
        # 0: bg, 1: un, 2: fg
        trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * alpha).long()
        if ignore_region is not None:
            trimap1[ignore_region] = 0
        trimap3 = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
        return trimap3.float()

    def preprocess(self, a, fg, bg, ignore_region=None, tri=None):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a
            scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            if bg is None:
                scaled_bgs = scaled_fgs
                scaled_imgs = scaled_fgs
            else:
                scaled_bgs = bg.flip([2]) * self.IMG_SCALE
                scaled_imgs = scaled_fgs * scaled_gts + scaled_bgs * (1. - scaled_gts)
            
            if tri is None:
                scaled_tris = self.make_trimap(scaled_gts, ignore_region)
            else:
                scaled_tris = tri
            imgs = scaled_imgs
        return scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, scaled_tris, imgs

    def _forward(self, imgs, tris, alpha, masks=None, og_shape=None):
        if self.stage == 1:
            batch_size, sample_length = imgs.shape[:2]
            num_object = torch.tensor([self.num_object]).to(torch.cuda.current_device())
            GT = tris.split(1, dim=0)                               # [1, S, C, H, W]
            FG = imgs.split(1, dim=0)                               # [1, S, C, H, W]
            
            if masks is not None:
                M = masks.squeeze(2).split(1, dim=1)
            E = []
            E_logits = []
            # we split batch here since the original code only supports b=1
            for b in range(batch_size):
                Fs = FG[b].split(1, dim=1)                          # [1, 1, C, H, W]
                GTs = GT[b].split(1, dim=1)                         # [1, 1, C, H, W]
                Es = [GTs[0].squeeze(1)] + [None] * (sample_length - 1) # [1, C, H, W]
                ELs = []
                for t in range(1, sample_length):
                    input_Es = Es[t-1]
                    # memorize
                    prev_key, prev_value = self.model(Fs[t-1].squeeze(1), input_Es, num_object)

                    if t-1 == 0: # 
                        this_keys, this_values = prev_key, prev_value # only prev memory
                    else:
                        this_keys = torch.cat([keys, prev_key], dim=3)
                        this_values = torch.cat([values, prev_value], dim=3)
                    
                    # segment
                    logit = self.model(Fs[t].squeeze(1), this_keys, this_values, num_object)
                    ELs.append(logit)
                    Es[t] = F.softmax(logit, dim=1)
                    
                    # update
                    keys, values = this_keys, this_values
                E.append(torch.cat(Es, dim=0))  # cat t
                E_logits.append(torch.cat(ELs, dim=0))

            pred = torch.stack(E, dim=0)  # stack b
            E_logits = [None] + list(torch.stack(E_logits).split(1, dim=1))
            GT = torch.argmax(tris, dim=2)
            # Loss & Vis
            losses = []
            for t in range(1, sample_length):
                gt = GT[:,t].squeeze(1)
                p = E_logits[t].squeeze(1)
                if og_shape is not None:
                    for b in range(batch_size):
                        h, w = og_shape[b]
                        gt[b, h:] = self.ignore_label
                        gt[b, :, w:] = self.ignore_label
                if masks is not None:
                    mask = M[t].squeeze(1)
                    gt = torch.where(mask == 0, torch.ones_like(gt) * self.ignore_label, gt)
                losses.append(self.LOSS(p, gt))
            loss = sum(losses) / float(len(losses))
            return pred, loss

    def _forward_single_step(self, img_q, img, tri, alpha, hid, memories=None):
        num_object = torch.tensor([self.num_object]).to(torch.cuda.current_device())
        # we split batch here since the original code only supports b=1
        if self.hdim > 0:
            Es = torch.cat([tri, alpha, hid], dim=1)
        else:
            Es = tri
        # memorize
        prev_key, prev_value = self.model(img, Es, num_object)

        # update
        if memories is None:
            memories = dict()
            memories['key'] = prev_key
            memories['val'] = prev_value
        else:
            memories['key'] = torch.cat([memories['key'], prev_key], dim=3)
            memories['val'] = torch.cat([memories['val'], prev_value], dim=3)

        # segment
        logit = self.model(img_q, memories['key'], memories['val'], num_object)
        return logit, memories

    def forward(self, a, fg, bg, ignore_region=None, tri=None, og_shape=None,
                single_step=False, hid=None, memories=None):
        if single_step:
            # fg: query frame (normalized between 0~1) [B, 3, H, W]
            # bg: prev frame (normalized between 0~1) [B, 3, H, W]
            # tri: prev trimap (normalized between 0~1) [B, 3, H, W]
            # a: prev alpha (normalized between 0~1) [B, 1, H, W]
            logit, memories = self._forward_single_step(fg, bg, tri, a, hid, memories=memories)
            return logit, memories
        else:
            scaled_imgs, _, _, scaled_gts, tris, imgs = self.preprocess(a, fg, bg, ignore_region=ignore_region, tri=tri)

            pred, loss = self._forward(imgs, tris, scaled_gts, og_shape=og_shape)
            
            return [loss, scaled_imgs, pred, tris, scaled_gts]


class FullModel_eval(FullModel):
    def _forward(self, imgs, tris, first_frame=False, masks=None, og_shape=None, save_memory=False, max_memory_num=2, memorize_gt=False):
        if self.stage == 1:
            num_object = torch.tensor([self.num_object]).to(torch.cuda.current_device())

            Fs = imgs
            
            if first_frame:
                Es = tris
                pred = Es
            else:
                logit = self.model(Fs, self.this_keys, self.this_values, num_object, memory_update=self.memory_update,)
                Es = F.softmax(logit, dim=1)
                pred = Es

            if save_memory and memorize_gt:
                Es = tris
                pred = tris
            prev_key, prev_value = self.model(Fs, Es, num_object)
            
            if max_memory_num == 0:
                if first_frame:
                    self.this_keys = prev_key
                    self.this_values = prev_value
            elif max_memory_num == 1:
                self.this_keys = prev_key
                self.this_values = prev_value
            else:
                if first_frame:
                    self.this_keys = prev_key
                    self.this_values = prev_value
                elif save_memory:
                    self.this_keys = torch.cat([self.this_keys, prev_key], dim=3)
                    self.this_values = torch.cat([self.this_values, prev_value], dim=3)
                else:
                    if self.this_keys.size(3) == 1:
                        self.this_keys = torch.cat([self.this_keys, prev_key], dim=3)
                        self.this_values = torch.cat([self.this_values, prev_value], dim=3)
                    else:
                        self.this_keys = torch.cat([self.this_keys[:,:,:,:-1], prev_key], dim=3)
                        self.this_values = torch.cat([self.this_values[:,:,:,:-1], prev_value], dim=3)

                if self.this_keys.size(3) > max_memory_num:
                    if memorize_gt:
                        self.this_keys = self.this_keys[:,:,:,1:]
                        self.this_values = self.this_values[:,:,:,1:]
                    else:
                        self.this_keys = torch.cat([self.this_keys[:,:,:,:1], self.this_keys[:,:,:,2:]], dim=3)
                        self.this_values = torch.cat([self.this_values[:,:,:,:1], self.this_values[:,:,:,2:]], dim=3)

            self.memory_update = save_memory

            return pred.unsqueeze(1), 0

    def _forward_memorize(self, img, tri, alpha, hid):
        num_object = torch.tensor([self.num_object]).to(torch.cuda.current_device())
        # we split batch here since the original code only supports b=1
        if self.hdim > 0:
            Es = torch.cat([tri, alpha, hid], dim=1)
        else:
            Es = tri
        # memorize
        prev_key, prev_value = self.model(img, Es, num_object)
        memories = {'key': prev_key,
                    'val': prev_value,
                    }
        return memories

    def _forward_segment(self, img_q, memories=None, memory_update=False):
        num_object = torch.tensor([self.num_object]).to(torch.cuda.current_device())
        # segment
        logit = self.model(img_q, memories['key'], memories['val'], num_object)
        return logit

    def forward(self, a, fg, bg, tri=None, first_frame=False, og_shape=None,
                memorize=False, segment=False, memories=None, hid=None,
                save_memory=False, max_memory_num=2, memory_update=False,
                memorize_gt=False,):
        if memorize:
            # fg: query frame (normalized between 0~1) [B, 3, H, W]
            # bg: prev frame (normalized between 0~1) [B, 3, H, W]
            # tri: prev trimap (normalized between 0~1) [B, 3, H, W]
            # a: prev alpha (normalized between 0~1) [B, 1, H, W]
            memories = self._forward_memorize(bg, tri, a, hid)
            return memories
        elif segment:
            # fg: query frame (normalized between 0~1) [B, 3, H, W]
            # bg: prev frame (normalized between 0~1) [B, 3, H, W]
            # tri: prev trimap (normalized between 0~1) [B, 3, H, W]
            # a: prev alpha (normalized between 0~1) [B, 1, H, W]
            logit = self._forward_segment(fg, memories=memories)
            return logit
        else:
            scaled_imgs, _, _, scaled_gts, tris, imgs = self.preprocess(a, fg, bg)
            if tri is not None:
                tris = tri
            imgs_fw_HR = imgs.squeeze(0)
            tris_fw = tris.squeeze(0)
            _, _, H, W = imgs_fw_HR.shape
            
            imgs_fw = imgs_fw_HR

            pred, loss = self._forward(imgs_fw, tris_fw, first_frame=first_frame, og_shape=og_shape, save_memory=save_memory, max_memory_num=max_memory_num, memorize_gt=memorize_gt,)

            if first_frame:
                pred = tris
            
            return [loss,
                    scaled_imgs, pred, tris, scaled_gts]
