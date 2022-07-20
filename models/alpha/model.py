import torch
import torch.nn.functional as F
from torch import nn

from models.alpha.FBA.models import FBA

import utils.loss_func as L
from utils.utils import trimap_transform
from models.alpha.common import pad_divide_by

class FullModel(nn.Module):
    FBA_LOSS_NORMALIZE = True
    FBA_L_ATT_MULTIPLIER = 1

    def __init__(self, dilate_kernel=None, eps=0,
                 trimap=None,
                 stage=1,
                 ):
        super(FullModel, self).__init__()

        self.stage = stage
        self.refinement = True if self.stage > 2 else False
        

        self.DILATION_KERNEL = dilate_kernel
        self.EPS = eps
        self.IMG_SCALE = 1./255
        self.register_buffer('IMG_MEAN', torch.tensor([0.485, 0.456, 0.406]).reshape([1, 1, 3, 1, 1]).float())
        self.register_buffer('IMG_STD', torch.tensor([0.229, 0.224, 0.225]).reshape([1, 1, 3, 1, 1]).float())
        self.memory_update = False
        
        self.NET = FBA(refinement=self.refinement)

        self.LAPLOSS = L.LapLoss()
        self.TRIMAP_CHANNEL = 8

        self.trimap = trimap
        self.LOSS_TRIMAP = nn.CrossEntropyLoss()

    def make_trimap(self, tri):
        tri = tri.float()
        scaled_tris = tri.max(dim=2)[1].unsqueeze(2).float()*0.5 #tri.float()# * self.IMG_SCALE
        trimask = ((scaled_tris > 0) & (scaled_tris < 1))
        if self.TRIMAP_CHANNEL == 3:
            scaled_tris = tri
        elif self.TRIMAP_CHANNEL == 8:
            trimap2f = (scaled_tris == 1).float()
            trimap2b = (scaled_tris == 0).float()
            trimap2 = torch.cat([trimap2b, trimap2f], dim=2)    # [B, S, 2, H, W]
            transformed_trimap = trimap_transform(trimap2)      # [B, S, 6, H, W]
            trimap2_soft = torch.stack([tri[:,:,0], tri[:,:,2]], dim=2)
            scaled_tris = torch.cat([transformed_trimap, trimap2_soft], dim=2).float()
        return scaled_tris, trimask.float()

    def preprocess(self, a, fg, bg, ignore_region=None, tri=None):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a# * self.IMG_SCALE
            scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            scaled_bgs = bg.flip([2]) * self.IMG_SCALE
            scaled_imgs = scaled_fgs * scaled_gts + scaled_bgs * (1. - scaled_gts)
            scaled_tris, trimasks = self.make_trimap(tri)
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)#.split(1, dim=1)
        return scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, scaled_tris, trimasks, imgs

    def single_image_loss(self, preds, trimasks, \
        scaled_gts, scaled_fgs, scaled_bgs, scaled_imgs, start, end):
        L_alpha, L_comp, L_grad = [], [], []
        sample_length = preds.shape[1]
        alphas, comps = [None] * sample_length, [None] * sample_length
        for c in range(start, end):
            c_gt = scaled_gts[:, c, ...]
            c_refine = preds[:, c, ...]
            alphas[c] = c_refine
            
            c_comp = scaled_fgs[:, c, ...] * c_refine + scaled_bgs[:, c, ...] * (1. - c_refine)
            c_img = scaled_imgs[:, c, ...]
            comps[c] = c_comp
            L_alpha.append(L.L1_mask(c_refine, c_gt))
            L_comp.append(L.L1_mask(c_comp, c_img))
            L_grad.append(L.L1_grad(c_refine, c_gt))
        L_comp = sum(L_comp) / float(len(L_comp))
        L_grad = sum(L_grad) / float(len(L_grad))
        L_alpha = sum(L_alpha) / float(len(L_alpha))
        for i in range(start):
            comps[i] = torch.zeros_like(comps[start])
            comps[-i-1] = torch.zeros_like(comps[start])
            alphas[i] = torch.zeros_like(alphas[start])
            alphas[-i-1] = torch.zeros_like(alphas[start])
        comps = torch.stack(comps, dim=1).clamp(0, 1)
        alphas = torch.stack(alphas, dim=1).clamp(0, 1)


        if (end - start) > 1:
            L_a_tc = F.mse_loss(alphas[:, 1:] - alphas[:, :-1], scaled_gts[:, 1:] - scaled_gts[:, :-1])
            L_tc = L_a_tc
            L_grad = L_grad + L_tc

        return L_alpha, L_comp, L_grad, alphas, comps

    def fba_single_image_loss(self, preds, trimasks, \
        scaled_gts, scaled_fgs, scaled_bgs, scaled_imgs, start, end,
        normalize):
        # Since FBA also outputs F and B...
        # preds [B, S, 7, H, W]
        sample_length = preds.shape[1]
        alpha = preds[:, :, :1, ...]
        predF = preds[:, :, 1:4, ...]
        predB = preds[:, :, 4:, ...]
        L_alpha_comp, L_lap, L_grad = [], [], []
        alphas, comps, Fs, Bs = [None] * sample_length, [None] * sample_length, \
                                [None] * sample_length, [None] * sample_length
        
        for c in range(start, end):
            c_gt = scaled_gts[:, c, ...]
            c_trimask = trimasks[:, c, ...]
            # c_refine = torch.where(c_trimask.bool(), alpha[:, c, ...], c_gt)
            c_refine = alpha[:, c, ...]
            c_img = scaled_imgs[:, c, ...]
            #c_fgmask = ((c_trimask == 1) + (c_gt == 1)).bool().repeat(1, 3, 1, 1)
            #c_bgmask = ((c_trimask == 1) + (c_gt == 0)).bool().repeat(1, 3, 1, 1)
            # c_F = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
            #                   predF[:, c, ...], scaled_fgs[:, c, ...])

            c_F = torch.where((c_trimask.bool() & (c_gt > 0)).repeat(1, 3, 1, 1), \
                              predF[:, c, ...], scaled_fgs[:, c, ...])
            c_B = torch.where(c_trimask.bool().repeat(1, 3, 1, 1), \
                              predB[:, c, ...], scaled_bgs[:, c, ...])

            # c_F = torch.where((c_gt > 0).repeat(1, 3, 1, 1), \
            #                   predF[:, c, ...], scaled_fgs[:, c, ...])
            # c_B = torch.where((c_trimask.bool() | (c_gt > 0)).repeat(1, 3, 1, 1), \
            #                   predB[:, c, ...], scaled_bgs[:, c, ...])
            
            alphas[c] = c_refine
            comps[c] = c_F * c_refine + c_B * (1. - c_refine)
            Fs[c] = c_F
            Bs[c] = c_B
            
            # There's no mean op in FBA paper, so we'll only sum (normalize=False)
            # L1 and comp related losses
            L_a1 = L.L1_mask(c_refine, c_gt, normalize=normalize)
            ac = c_F * c_gt + c_B * (1. - c_gt)
            L_ac = L.L1_mask(ac, c_img, normalize=normalize)
            FBc = scaled_fgs[:, c, ...] * c_refine + scaled_bgs[:, c, ...] * (1. - c_refine)
            L_FBc = L.L1_mask(FBc, c_img, normalize=normalize)
            L_FB1 = L.L1_mask(c_F, scaled_fgs[:, c, ...], normalize=normalize) + \
                    L.L1_mask(c_B, scaled_bgs[:, c, ...], normalize=normalize)
            L_alpha_comp.append(L_a1 + L_ac + 0.25 * (L_FBc + L_FB1))
            
            # gradient related losses
            L_ag = L.L1_grad(c_refine, c_gt, normalize=normalize)
            #L_grad.append(L_ag)
            L_FBexcl = L.exclusion_loss(c_F, c_B, level=3, normalize=normalize)
            L_grad.append(L_ag + 0.25 * L_FBexcl)

            # Laplacian loss
            L_a_lap = self.LAPLOSS(c_refine, c_gt, normalize=normalize)
            L_F_lap = self.LAPLOSS(c_F, scaled_fgs[:, c, ...], normalize=normalize)
            L_B_lap = self.LAPLOSS(c_B, scaled_bgs[:, c, ...], normalize=normalize)
            L_lap.append(L_a_lap + 0.25 * (L_F_lap + L_B_lap))
        L_alpha_comp = sum(L_alpha_comp) / float(len(L_alpha_comp))
        L_grad = sum(L_grad) / float(len(L_grad))
        L_lap = sum(L_lap) / float(len(L_lap))

        for i in range(start):
            comps[i] = torch.zeros_like(comps[start])
            comps[-i-1] = torch.zeros_like(comps[start])
            alphas[i] = torch.zeros_like(alphas[start])
            alphas[-i-1] = torch.zeros_like(alphas[start])
            Fs[i] = torch.zeros_like(Fs[start])
            Fs[-i-1] = torch.zeros_like(Fs[start])
            Bs[i] = torch.zeros_like(Bs[start])
            Bs[-i-1] = torch.zeros_like(Bs[start])
        comps = torch.stack(comps, dim=1)
        Fs = torch.stack(Fs, dim=1)
        Bs = torch.stack(Bs, dim=1)
        alphas = torch.stack(alphas, dim=1)

        if (end - start) > 1:
            L_a_tc = F.mse_loss(alphas[:, 1:] - alphas[:, :-1], scaled_gts[:, 1:] - scaled_gts[:, :-1])
            L_F_tc = F.mse_loss(Fs[:, 1:] - Fs[:, :-1], scaled_fgs[:, 1:] - scaled_fgs[:, :-1])
            L_B_tc = F.mse_loss(Bs[:, 1:] - Bs[:, :-1], scaled_bgs[:, 1:] - scaled_bgs[:, :-1])
            L_tc = L_a_tc + 0.25 * (L_F_tc + L_B_tc)
            L_grad = L_grad + L_tc

        return L_alpha_comp, L_lap, L_grad, alphas, comps, Fs, Bs

    def forward(self, a, fg, bg, ignore_region=None, tri=None):
        B, sample_length, _, H, W = a.shape
        c = sample_length // 2
        start, end = 0, sample_length
        scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, tris, trimasks, imgs = self.preprocess(a, fg, bg, ignore_region=ignore_region, tri=tri)

        preds_alpha = [None] * sample_length
        preds_alpha_refine = [None] * sample_length
        preds_trimap = [None] * sample_length
        preds_trimap_refine = [None] * sample_length
        logit_trimap = [None] * (sample_length-1)
        logit_trimap_refine = [None] * (sample_length)

        _imgs = imgs
        _scaled_imgs = scaled_imgs
        _tri = tri

        memories = None
        preds_trimap[0] = _tri[:,0]
        preds_trimap_refine[0] = _tri[:,0]
        key, value = torch.zeros([1]), torch.zeros([1])
        for t in range(sample_length):
            # inputs = list(torch.cat([imgs, tris], dim=2).split(1, dim=1))
            scaled_tris, _ = self.make_trimap(preds_trimap[t].unsqueeze(1))
            scaled_tris = scaled_tris.squeeze(1)
            
            inputs = torch.cat([_imgs[:,t], scaled_tris], dim=1)    
            extras = [_scaled_imgs[:, t], scaled_tris[:, -2:]]
            preds_alpha[t], hid, preds_alpha_refine[t], _logit_trimap_refine = self.NET(inputs, extras=extras)
            
            if self.refinement:
                logit_trimap_refine[t] = _logit_trimap_refine
                if t > 0:
                    preds_trimap_refine[t] = F.softmax(_logit_trimap_refine, dim=1)
            
            if t < (sample_length-1):
                if self.trimap is None:
                    logit_trimap[t] = _tri[:,t+1]
                    preds_trimap[t+1] = _tri[:,t+1]
                else:
                    if self.refinement:
                        input_alpha = preds_alpha_refine[t][:,:1]
                        input_trimap = preds_trimap_refine[t]
                    else:
                        input_alpha = preds_alpha[t][:,:1]
                        input_trimap = preds_trimap[t]
                    
                    _logit_trimap, memories = self.trimap(a=input_alpha,
                                                        fg=_scaled_imgs[:, t+1],
                                                        bg=_scaled_imgs[:, t],
                                                        tri=input_trimap,
                                                        single_step=True,
                                                        hid=hid,
                                                        memories=memories,)
                    logit_trimap[t] = _logit_trimap
                    preds_trimap[t+1] = F.softmax(_logit_trimap, dim=1)
        preds_alpha = torch.stack(preds_alpha, dim=1)
        if self.refinement:
            preds_alpha_refine = torch.stack(preds_alpha_refine, dim=1)
        if self.refinement:
            preds_trimap = torch.stack(preds_trimap_refine, dim=1)
            logit_trimap_refine = torch.stack(logit_trimap_refine, dim=1)
        else:
            preds_trimap = torch.stack(preds_trimap, dim=1)
        if self.trimap is not None:
            logit_trimap = torch.stack(logit_trimap, dim=1)
            
        loss_inputs = (preds_alpha, trimasks, scaled_gts, \
            scaled_fgs, scaled_bgs, scaled_imgs, \
            start, end)
            
        # L_alpha_comp, L_lap, L_grad
        LOSS1 = self.fba_single_image_loss(*loss_inputs, normalize=self.FBA_LOSS_NORMALIZE)

        if self.refinement:
            refine_loss_inputs = (preds_alpha_refine, trimasks, scaled_gts, \
                scaled_fgs, scaled_bgs, scaled_imgs, \
                start, end)
                
            # L_alpha_comp, L_lap, L_grad
            LOSS2 = \
                self.fba_single_image_loss(*refine_loss_inputs, normalize=self.FBA_LOSS_NORMALIZE)
            loss1 = LOSS1[0] + LOSS2[0]
            loss2 = LOSS1[1] + LOSS2[1]
            loss3 = LOSS1[2] + LOSS2[2]
            alphas = LOSS2[3]
            comps = LOSS2[4]
            Fs = LOSS2[5]
            Bs = LOSS2[6]
        else:
            # L_alpha_comp, L_lap, L_grad
            loss1 = LOSS1[0]
            loss2 = LOSS1[1]
            loss3 = LOSS1[2]
            alphas = LOSS1[3]
            comps = LOSS1[4]
            Fs = LOSS1[5]
            Bs = LOSS1[6]


        if self.trimap is not None:
            GT_TRIMAP = torch.argmax(tri[:,1:], dim=2).view(-1, H, W)
            loss_trimap = self.LOSS_TRIMAP(logit_trimap.view(-1, 3, H, W), GT_TRIMAP)
            if self.refinement:
                GT_TRIMAP = torch.argmax(tri, dim=2).view(-1, H, W)
                loss_trimap = loss_trimap + self.LOSS_TRIMAP(logit_trimap_refine.view(-1, 3, H, W), GT_TRIMAP)
        else:
            loss_trimap = torch.zeros([1]).to(torch.cuda.current_device())



        with torch.no_grad():
            if self.TRIMAP_CHANNEL != 1:
                tris_vis = torch.where(trimasks.bool(), \
                    torch.ones_like(scaled_gts)*128*self.IMG_SCALE, \
                    scaled_gts)
            else:
                tris_vis = tris


        return [loss1, loss2, loss3, loss_trimap,       # Loss
                scaled_imgs, tris_vis, alphas, comps,   # Vis
                scaled_gts, Fs, Bs,
                preds_trimap]

class EvalModel(FullModel):
    def preprocess(self, img, tri):
        # Data preprocess
        with torch.no_grad():
            scaled_imgs = img.float().flip([2]) * self.IMG_SCALE
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)
            scaled_tris = tri.float() * self.IMG_SCALE
            trimask = ((scaled_tris > 0) & (scaled_tris < 1))
            if self.DILATION_KERNEL is not None:
                trimask = trimask.float().split(1)    # trimasks: B * [S, C, H, W]
                b = len(trimask)
                trimasks = [None] * b
                for i in range(b):
                    trimasks[i] = F.max_pool2d(trimask[i].squeeze(0), \
                        kernel_size=self.DILATION_KERNEL*2+1, stride=1, padding=self.DILATION_KERNEL)
                trimask = torch.stack(trimasks).bool()
            if self.TRIMAP_CHANNEL == 3:
                trimap1 = torch.where(trimask, torch.ones_like(scaled_tris), 2 * scaled_tris).long()
                scaled_tris = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
            elif self.TRIMAP_CHANNEL == 8:
                trimap2f = (scaled_tris == 1).float()
                trimap2b = (scaled_tris == 0).float()
                trimap2 = torch.cat([trimap2b, trimap2f], dim=2)    # [B, S, 2, H, W]
                transformed_trimap = trimap_transform(trimap2)      # [B, S, 6, H, W]
                scaled_tris = torch.cat([transformed_trimap, trimap2], dim=2).float()
        return scaled_imgs, scaled_tris, trimask.float(), imgs


    def make_trimap_gt(self, alpha, trimap3=None):
        if trimap3 is None:
            b = alpha.shape[0]
            alpha = torch.where(alpha < self.EPS, torch.zeros_like(alpha), alpha)
            alpha = torch.where(alpha > 1 - self.EPS, torch.ones_like(alpha), alpha)
            trimasks = ((alpha > 0) & (alpha < 1.)).float().split(1)    # trimasks: B * [S, C, H, W]
            trimaps = [None] * b
            for i in range(b):
                # trimap width: 1 - 51
                kernel_rad = int(torch.randint(0, 26, size=())) \
                    if self.DILATION_KERNEL is None else self.DILATION_KERNEL
                trimaps[i] = F.max_pool2d(trimasks[i].squeeze(0), kernel_size=kernel_rad*2+1, stride=1, padding=kernel_rad)
            trimap = torch.stack(trimaps)
        else:
            _, trimap_ = trimap3.max(dim=2)
            trimap = (trimap_ == 1).unsqueeze(2).float()
            alpha = trimap_ / 2.

        # 0: bg, 1: un, 2: fg
        trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * alpha).long()
        trimap3 = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3).float()

        if self.TRIMAP_CHANNEL == 1:
            trimap1 = torch.where(trimap > 0.5, 128.*torch.ones_like(alpha)*self.IMG_SCALE, alpha)
            return trimap1, trimap, trimap3
        elif self.TRIMAP_CHANNEL == 3:
            # # 0: bg, 1: un, 2: fg
            # trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2 * alpha).long()
            # trimap3 = F.one_hot(trimap1.squeeze(2), num_classes=3).permute(0, 1, 4, 2, 3)
            return trimap3.float(), trimap, trimap3
        elif self.TRIMAP_CHANNEL == 8:
            trimap1 = torch.where(trimap > 0.5, 255*torch.ones_like(alpha), alpha)
            trimap2f = (trimap1 == 1).float()
            trimap2b = (trimap1 == 0).float()
            trimap2 = torch.cat([trimap2b, trimap2f], dim=2)    # [B, S, 2, H, W]
            transformed_trimap = trimap_transform(trimap2)      # [B, S, 6, H, W]
            return torch.cat([transformed_trimap, trimap2], dim=2).float(), trimap, trimap3

    def preprocess_gt(self, a, fg, bg):
        # Data preprocess
        with torch.no_grad():
            scaled_gts = a# * self.IMG_SCALE
            scaled_fgs = fg.flip([2]) * self.IMG_SCALE
            scaled_bgs = bg.flip([2]) * self.IMG_SCALE
            scaled_imgs = scaled_fgs * scaled_gts + scaled_bgs * (1. - scaled_gts)
            scaled_tris, trimasks, trimap3 = self.make_trimap_gt(scaled_gts)
            imgs = ((scaled_imgs - self.IMG_MEAN) / self.IMG_STD)#.split(1, dim=1)
        return scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, scaled_tris, trimasks, trimap3, imgs

    def forward(self, a, fg, bg, tri=None, tri_gt=None, first_frame=False, last_frame=False,
                memorize=False, max_memory_num=2, large_input=False):
        scaled_imgs, scaled_fgs, scaled_bgs, scaled_gts, tris_gt, trimasks_gt, trimap3_gt, imgs = self.preprocess_gt(a, fg, bg)

        if tri is not None:
            tri = tri.flip([2]) * self.IMG_SCALE
        elif tri_gt is not None:
            tri = tri_gt
            scaled_tris, trimasks, trimap3_gt = self.make_trimap_gt(None, trimap3=tri)
        else:
            tri = trimap3_gt
        tri_gt = trimap3_gt
        
        scaled_imgs_HR = scaled_imgs.squeeze(0)
        tri_ = tri.squeeze(0)
        _, _, H, W = scaled_imgs_HR.shape

        (scaled_imgs_HR), pad = pad_divide_by([scaled_imgs_HR], int(32), (scaled_imgs_HR.size()[-2], scaled_imgs_HR.size()[-1]))
        if sum(pad) > 0:
            tri_ = torch.cat((F.pad(tri_[:, :1], pad, value=1.), F.pad(tri_[:, 1:], pad, value=0.)), dim=1)

        scaled_imgs_ = scaled_imgs_HR

        imgs_ = (scaled_imgs_ - self.IMG_MEAN.squeeze(0)) / self.IMG_STD.squeeze(0)

        scaled_tris_, _ = self.make_trimap(tri_.unsqueeze(0))
        scaled_tris_ = scaled_tris_.squeeze(0)

        if self.trimap is None:
            pass
        else:
            scaled_imgs_for_tri = scaled_imgs_

            # segment 
            if first_frame:
                self.first_memories = None
                self.memories = {'key': None,
                                'val': None,
                                }
                preds_trimap = tri_
            else:
                _logit_trimap = self.trimap(a=None,
                                            fg=scaled_imgs_for_tri,
                                            bg=None,
                                            tri=None,
                                            memories=self.memories,
                                            segment=True,
                                            memory_update=self.memory_update)

                preds_trimap = F.softmax(_logit_trimap, dim=1)

                scaled_tris_, _ = self.make_trimap(preds_trimap.unsqueeze(0))
                scaled_tris_ = scaled_tris_.squeeze(0)
        
        inputs = torch.cat([imgs_, scaled_tris_], dim=1)
        extras = [scaled_imgs_, scaled_tris_[:, -2:]]
        if large_input:
            torch.cuda.empty_cache()
        preds_alpha, hid, preds_alpha_refine, _logit_trimap_refine = self.NET(inputs, extras=extras)

        if self.refinement:
            preds_alpha = preds_alpha_refine[:, :1, ...]
        else:
            preds_alpha = preds_alpha[:, :1, ...]

        if self.trimap is None:
            preds_trimap = tri
        else:
            if self.refinement:
                preds_trimap = F.softmax(_logit_trimap_refine, dim=1)
            if not last_frame:
                # memorize
                preds_alpha_for_tri = preds_alpha
                hid_for_tri = hid

                memories = self.trimap(a=preds_alpha_for_tri,
                                    fg=None,
                                    bg=scaled_imgs_for_tri,
                                    tri=preds_trimap,
                                    memorize=True,
                                    hid=hid_for_tri)
                if max_memory_num == 0:
                    if first_frame:
                        self.memories = memories
                elif max_memory_num == 1:
                    self.memories = memories
                else:
                    if first_frame:
                        self.memories = memories
                    elif memorize:
                        self.memories['key'] = torch.cat([self.memories['key'], memories['key']], dim=3)
                        self.memories['val'] = torch.cat([self.memories['val'], memories['val']], dim=3)
                    else:
                        if self.memories['key'].size(3) == 1:
                            self.memories['key'] = torch.cat([self.memories['key'], memories['key']], dim=3)
                            self.memories['val'] = torch.cat([self.memories['val'], memories['val']], dim=3)
                        else:
                            self.memories['key'] = torch.cat([self.memories['key'][:,:,:,:-1], memories['key']], dim=3)
                            self.memories['val'] = torch.cat([self.memories['val'][:,:,:,:-1], memories['val']], dim=3)

                    if self.memories['key'].size(3) > max_memory_num:
                        self.memories['key'] = torch.cat([self.memories['key'][:,:,:,:1], self.memories['key'][:,:,:,2:]], dim=3)
                        self.memories['val'] = torch.cat([self.memories['val'][:,:,:,:1], self.memories['val'][:,:,:,2:]], dim=3)
            
            if pad[2]+pad[3] > 0:
                preds_trimap = preds_trimap[:,:,pad[2]:-pad[3],:]
            if pad[0]+pad[1] > 0:
                preds_trimap = preds_trimap[:,:,:,pad[0]:-pad[1]]

            preds_trimap = preds_trimap.unsqueeze(0)


        if pad[2]+pad[3] > 0:
            preds_alpha = preds_alpha[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            preds_alpha = preds_alpha[:,:,:,pad[0]:-pad[1]]

        preds_alpha = preds_alpha.unsqueeze(0)

        self.memory_update = memorize

        return scaled_imgs, preds_trimap, tri_gt, preds_alpha, scaled_gts
