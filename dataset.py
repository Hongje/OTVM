import os
import json
import random
import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from imgaug import parameters as iap
from six.moves import range
import tqdm

try:
    import pickle5 as pickle
except:
    import pickle
import albumentations as A
from skimage import exposure

"""
affine_transforms.py
"""
def channel_shift(xs, intensity, channel_axis):
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = np.rollaxis(x, channel_axis, 0)
            min_x, max_x = np.min(x), np.max(x)
            channel_images = [np.clip(x_channel + intensity, min_x, max_x)
                            for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
            ys.append(x)

        else:
            ys.append(x)

    return ys

def apply_transforms_cv(xs, M):
    """Apply the image transformation specified by a matrix.
    """
    dsize = (np.int(xs[0].shape[1]), np.int(xs[0].shape[0]))

    aff = M[:2, :2]
    off = M[:2, 2]
    cvM = np.zeros_like(M[:2, :])
    # cvM[:2,:2] = aff
    cvM[:2,:2] = np.flipud(np.fliplr(aff))
    # cvM[:2,:2] = np.transpose(aff)
    cvM[:2, 2] = np.flip(off, axis=0)
    ys = []
    for x in xs:
        if x.ndim == 3: # image
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_LINEAR)
            ys.append(x)

        else: # mask
            x = cv2.warpAffine(x, cvM, dsize, flags=cv2.INTER_NEAREST)
            ys.append(x)

    
    return ys


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def flip_axis(xs, axis):
    ys = []
    for x in xs:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        ys.append(x)

    return ys

def random_transform(xs, rnd,
                     rt=False, # rotation
                     hs=False, # height_shift
                     ws=False, # width_shift
                     sh=False, # shear
                     zm=[1,1], # zoom
                     sc=[1,1],
                     cs=False, # channel shift
                     hf=False): # horizontal flip
                    
    """Randomly augment a single image tensor.
    """
    # x is a single image, so it doesn't have image number at index 0
    img_row_axis = 0
    img_col_axis = 1
    img_channel_axis = 2
    h, w = xs[0].shape[img_row_axis], xs[0].shape[img_col_axis]

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rt:
        theta = np.pi / 180 * rnd.uniform(-rt, rt)
    else:
        theta = 0

    if hs:
        tx = rnd.uniform(-hs, hs) * h
    else:
        tx = 0 
 
    if ws:
        ty = rnd.uniform(-ws, ws) * w
    else:
        ty = 0

    if sh:
        shear = np.pi / 180 * rnd.uniform(-sh, sh)
    else:
        shear = 0

    if zm[0] == 1 and zm[1] == 1:
        zx, zy = 1, 1
    else:
        zx = rnd.uniform(zm[0], zm[1])
        zy = rnd.uniform(zm[0], zm[1])

    if sc[0] == 1 and sc[1] == 1:
        zx, zy = zx, zy
    else:
        s = rnd.uniform(sc[0], sc[1])
        zx = zx * s
        zy = zy * s

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix


    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                    [0, 1, ty],
                                    [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        if rnd.random() < 0.5:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
        else:
            shear_matrix = np.array([[np.cos(shear), 0, 0],
                                    [np.sin(shear), 1, 0],
                                    [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)

        xs = apply_transforms_cv(xs, transform_matrix)

        # plt.figure(1)
        # plt.subplot(2,1,1); plt.imshow(xs[0])
        # plt.subplot(2,1,2); plt.imshow(xs[1])
        # plt.show()


    if cs != 0:
        intensity = rnd.uniform(-cs, cs)
        xs = channel_shift(xs,
                            intensity,
                            img_channel_axis)
    
    if hf:
        if rnd.random() < 0.5:
            xs = flip_axis(xs, img_col_axis)

    return xs

def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def make_trimap(rnd, alpha, eps=0., dilation_kernel=None, close_first=False, ignore_region=None):
    # b = alpha.shape[0]
    alpha = torch.where(alpha < eps, torch.zeros_like(alpha), alpha)
    alpha = torch.where(alpha > 1 - eps, torch.ones_like(alpha), alpha)
    trimap = ((alpha > 0) & (alpha < 1.)).float()
    kernel_rad_close = 0
    if close_first:
        trimap_ori = trimap.clone()
        kernel_rad_close = rnd.randint(0,5)
        trimap = 1. - trimap
        if ignore_region is not None:
            trimap[ignore_region] = 0
        trimap = F.max_pool2d(trimap, kernel_size=kernel_rad_close*2+1, stride=1, padding=kernel_rad_close)
        trimap = 1. - trimap
        if ignore_region is not None:
            trimap[ignore_region] = 0
        trimap = F.max_pool2d(trimap, kernel_size=kernel_rad_close*2+1, stride=1, padding=kernel_rad_close)
    kernel_rad = dilation_kernel
    trimap = F.max_pool2d(trimap, kernel_size=kernel_rad*2+1, stride=1, padding=kernel_rad)

    if close_first:
        trimap = trimap + trimap_ori

    # 0: bg, 1: un, 2: fg
    trimap1 = torch.where(trimap > 0.5, torch.ones_like(alpha), 2. * (alpha>0.5)).long()
    if ignore_region is not None:
        trimap1[ignore_region] = 0
        alpha[ignore_region] = 0
    trimap3 = F.one_hot(trimap1.squeeze(1), num_classes=3).permute(0, 3, 1, 2)
    return trimap3.float(), alpha


class VideoMatting108_Train(torch.utils.data.Dataset):
    def __init__(self, data_root, image_shape,
                 mode='train',
                 use_subset=False,
                 sample_length=3,
                 max_skip=75,
                 do_affine=0.5,
                 do_time_flip=0.5,
                 do_histogram_matching=0.3,
                 do_gamma_aug=0.3,
                 do_jpeg_aug=0.3,
                 do_gaussian_aug=0.3,
                 do_motion_aug=0.3,):
        self.mode = mode
        self.use_subset = use_subset
        self.sample_length = sample_length
        self.max_skip = max_skip
        self.do_affine = do_affine
        self.do_time_flip = do_time_flip
        self.do_histogram_matching = do_histogram_matching
        self.do_gamma_aug = do_gamma_aug
        self.do_jpeg_aug = do_jpeg_aug
        self.do_gaussian_aug = do_gaussian_aug
        self.do_motion_aug = do_motion_aug
        assert self.mode in ['train', 'val']
        self.root = data_root
        self.image_shape = list(image_shape)
        
        self.data_root = dict()
        self.FG = list()
        self.BG = list()
        self.Alpha = list()
        
        self.data_root['V108'] = os.path.join(self.root, 'VideoMatting108')
        setname = '{}_videos_subset.txt' if self.use_subset else '{}_videos.txt'
        setname = setname.format(self.mode)
        with open(os.path.join(self.data_root['V108'], 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)
        with open(os.path.join(self.data_root['V108'], setname), 'r') as f:
            self.FG, self.BG = self.parse_VideoMatting108(f, self.frame_corr, FG_FOLDER='FG_done', BG_FOLDER='BG_done2')
        self.FG_len = len(self.FG)
        self.BG_len = len(self.BG)
        
        self.pixel_aug_gamma = iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5))
        self.pixel_aug_gaussian = iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))
        self.jpeg_aug = iaa.JpegCompression(compression=(20, 80)) 
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3,50))
        

        self.EdgeFilter = nn.Conv2d(1, 2, kernel_size=(3,3), stride=1, bias=False) # No Padding
        self.EdgeFilter.weight = nn.Parameter(torch.Tensor([[[[1., 0., -1.,],
                                                                [2., 0,  -2.,],
                                                                [1., 0., -1.]]],
                                                            [[[ 1.,  2.,  1.,],
                                                                [ 0.,  0,   0.,],
                                                                [-1., -2., -1.]]]]))
        for param in self.EdgeFilter.parameters():
            param.requires_grad = False


    def __len__(self):
        return self.FG_len

    def parse_VideoMatting108(self, f, frame_corr, FG_FOLDER, BG_FOLDER):
        FG = list()
        BG = list()
        print('parse VideoMatting108 dataset')
        for v in tqdm.tqdm(f):
            FG_path_current = list()
            BG_path_current = list()
            v = v.strip()
            fns = [k for k in sorted(self.frame_corr.keys()) if os.path.dirname(k) == v]
            for i in range(len(fns)):
                FG_path_current.append(os.path.join(FG_FOLDER, fns[i]))
                BG_path_current.append(os.path.join(BG_FOLDER, frame_corr[fns[i]]))
            FG.append(['V108', FG_path_current])
            BG.append(['V108', BG_path_current])
            
        return FG, BG

    def random_crop(self, N_frames, N_masks, num_frames, rnd):
        real_size = N_frames[0].shape[:2]
        do_mask = N_masks is not None
        ## random transformations that both to be applied.
        min_scale = np.maximum(self.image_shape[0]/np.float(real_size[0]), self.image_shape[1]/np.float(real_size[1]))

        for t in range(100):
            scale = np.maximum(rnd.choice([1., 1./1.5, 1./2.]), min_scale+0.01)
            dsize = (np.int(real_size[1]*scale), np.int(real_size[0]*scale))

            _rz_N_frames = np.empty((num_frames, dsize[1], dsize[0], N_frames[0].shape[2]), dtype=np.float32)
            if do_mask:
                _rz_N_masks = np.empty((num_frames, dsize[1], dsize[0]), dtype=np.float32) 
            for f in range(num_frames):
                _rz_N_frames[f] = cv2.resize(N_frames[f], dsize=dsize, interpolation=cv2.INTER_LINEAR)
                if do_mask:
                    _rz_N_masks[f] = cv2.resize(N_masks[f], dsize=dsize, interpolation=cv2.INTER_LINEAR)
            rz_N_frames = _rz_N_frames
            if do_mask:
                rz_N_masks = _rz_N_masks
                np_in1 = None
            for tt in range(1000):
                cr_y = rnd.randint(0, _rz_N_frames.shape[1] - self.image_shape[0])
                cr_x = rnd.randint(0, _rz_N_frames.shape[2] - self.image_shape[1])
                if do_mask:
                    center_alpha_val = rz_N_masks[0, cr_y+int(self.image_shape[0]//2), cr_x+int(self.image_shape[1]//2)]
                    if (tt < 900) or (t < 90):
                        if (center_alpha_val > (0.2*255)) and (center_alpha_val < (0.8*255)):
                            crop_N_masks = rz_N_masks[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1]]
                            break
                    else:
                        if np_in1 is None:
                            np_in1 = np.sum((rz_N_masks[0] > (0.2*255)) & (rz_N_masks[0] < (0.8*255)))
                        crop_N_masks = rz_N_masks[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1]]
                        crop_N_masks_UR = (crop_N_masks[0] > (0.2*255)) & (crop_N_masks[0] < (0.8*255))
                        if (np.sum(crop_N_masks_UR) > 0.5*np_in1) or np.mean(crop_N_masks_UR) > 0.01/255.:
                            break
                else:
                    crop_N_masks = None
                    break

            if tt < 999:
                break
        crop_N_frames = rz_N_frames[:,cr_y:cr_y+self.image_shape[0], cr_x:cr_x+self.image_shape[1],:]
        
        return crop_N_frames, crop_N_masks, cr_y * (1.0 / scale), cr_x * (1.0 / scale)

    def sample_num_skip(self, sample_length, max_skip, rnd):
        skips = [0] + [rnd.randint(0, max_skip) for _ in range(sample_length-1)]
        com = [sum(skips[:i+1]) for i in range(len(skips))]
        return com

    def __getitem__(self, idx):
        info = dict()
        rnd = random.Random()
        
        data_FG, sample_FG = self.FG[idx]
        sample_FG_len = len(sample_FG)
        idx_BG = rnd.randint(0, self.BG_len-1)
        data_BG, sample_BG = self.BG[idx_BG]
        sample_BG_len = len(sample_BG)

        max_skip = self.max_skip
        
        ttr = 0
        while True:
            if ttr > 1000:
                return self.__getitem__(rnd.randint(0, self.__len__()-1))
            ttr += 1
            if ttr > 600:
                cum = self.sample_num_skip(self.sample_length, 0, rnd)
            else:
                cum = self.sample_num_skip(self.sample_length, max_skip, rnd)
            if (sample_FG_len-self.sample_length-cum[-1] > 1) and (sample_BG_len-self.sample_length-cum[-1] > 1):
                break
        info['cum'] = cum

        if self.mode == 'train' and rnd.uniform(0,1) < self.do_time_flip:
            sample_FG = sample_FG[::-1]
        N_st_FG = rnd.randint(0, sample_FG_len-self.sample_length-cum[-1])
        sample_FG = [sample_FG[N_st_FG+cum_] for cum_ in cum]
        if self.BG_len > 0:
            if self.mode == 'train' and rnd.uniform(0,1) < self.do_time_flip:
                sample_BG = sample_BG[::-1]
            N_st_BG = rnd.randint(0, sample_BG_len-self.sample_length-cum[-1])
            sample_BG = [sample_BG[N_st_BG+cum_] for cum_ in cum]

        fg, bg, a = [None] * self.sample_length, [None] * self.sample_length, [None] * self.sample_length

        # img I/O
        # FG & Alpha
        data_root_FG = self.data_root[data_FG]
        for i in range(self.sample_length):
            _f = cv2.imread(os.path.join(data_root_FG, sample_FG[i]), cv2.IMREAD_UNCHANGED)
            fg[i] = _f[..., :-1]
            a[i] = _f[..., -1]

        if a[0].sum() < 1:
            return self.__getitem__(rnd.randint(0, self.__len__()-1))
            
        # BG
        data_root_BG = self.data_root[data_BG]
        for i in range(self.sample_length):
            bgp = os.path.join(data_root_BG, sample_BG[i])
            if not os.path.exists(bgp):
                bgp = os.path.splitext(bgp)[0]+'.png'
            bg[i] = cv2.imread(bgp, cv2.IMREAD_COLOR)



        for i in range(self.sample_length):
            fg[i] = np.float32(fg[i])
            a[i] = np.float32(a[i])
            bg[i] = np.float32(bg[i])
            
        fg, a, scr_y, scr_x = self.random_crop(fg, a, self.sample_length, rnd)
        if bg[0] is not None:
            bg, _, scr_y, scr_x = self.random_crop(bg, None, self.sample_length, rnd)

        # gamma augmentation
        if (rnd.uniform(0,1) < self.do_gamma_aug):
            fg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(fg_aug.augment_image(np.uint8(fg[i])))
        
        if (rnd.uniform(0,1) < self.do_gamma_aug) and (bg[0] is not None):
            bg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                bg[i] = np.float32(bg_aug.augment_image(np.uint8(bg[i])))

        if (rnd.uniform(0,1) < self.do_histogram_matching) and (bg[0] is not None):
            ratio = rnd.uniform(0,0.5)
            if rnd.uniform(0,1) < 0.05:
                bg_match = exposure.match_histograms(bg, fg, channel_axis=-1)
                bg = bg_match * ratio + bg * (1. - ratio)
            else:
                fg_match = exposure.match_histograms(fg, bg, channel_axis=-1)
                fg = fg_match * ratio + fg * (1. - ratio)
                    
        # random H flip 
        if rnd.randint(0,1) == 0:
            fg = _flip_axis(fg, 2)
            a = _flip_axis(a, 2)
        if rnd.randint(0,1) == 0 and (bg[0] is not None):
            bg = _flip_axis(bg, 2)


        # motion augmentation
        if (rnd.uniform(0,1) < self.do_motion_aug):
            if rnd.uniform(0,1) < 0.5 and (bg[0] is not None):
                N_cat = np.concatenate([fg, bg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                fg = N_cat_aug[..., :3]
                bg = N_cat_aug[..., 3:6]
                a = N_cat_aug[..., 6]
                fg = np.clip(fg, 0, 255)
                bg = np.clip(bg, 0, 255)
                a = np.clip(a, 0, 255)
            else:
                if rnd.uniform(0,1) < 0.9:
                    N_cat = np.concatenate([fg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    fg = N_cat_aug[..., :3]
                    a = N_cat_aug[..., 3]
                    fg = np.clip(fg, 0, 255)
                    a = np.clip(a, 0, 255)
                if rnd.uniform(0,1) < 0.3 and (bg[0] is not None):
                    N_cat = bg # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    bg = N_cat_aug
                    bg = np.clip(bg, 0, 255)

        # augmentation
        if (rnd.uniform(0,1) < self.do_gaussian_aug):
            aug = self.pixel_aug_gaussian.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                if bg[0] is not None:
                    bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))
        if (rnd.uniform(0,1) < self.do_jpeg_aug):
            aug = self.jpeg_aug.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                a[i] = np.float32(aug.augment_image(np.uint8(a[i])))
                if bg[0] is not None:
                    bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))

        # random affine
        ignore_region = np.ones_like(a)
        if rnd.uniform(0,1) < self.do_affine:
            if bg[0] is not None:
                list_FM = list(fg) + list(a) + list(ignore_region) + list(bg)
            else:
                list_FM = list(fg) + list(a) + list(ignore_region)
            list_trans_FM = random_transform(list_FM, rnd, rt=10, sh=5, zm=[0.95,1.05], sc= [1, 1], cs=0.03*255., hf=False)
            fg = np.stack(list_trans_FM[:self.sample_length], axis=0)
            a = np.stack(list_trans_FM[self.sample_length:int(self.sample_length*2)], axis=0)
            ignore_region = np.stack(list_trans_FM[int(self.sample_length*2):int(self.sample_length*3)], axis=0)
            if bg[0] is not None:
                bg = np.stack(list_trans_FM[int(self.sample_length*3):int(self.sample_length*4)], axis=0)
        
        a = a / 255.

        fg = torch.from_numpy(np.transpose(fg, (0, 3, 1, 2)).copy()).float()
        if bg[0] is not None:
            bg = torch.from_numpy(np.transpose(bg, (0, 3, 1, 2)).copy()).float()
        else:
            bg = fg.clone()
        a = torch.from_numpy(a.copy()).unsqueeze(1).float()
        ignore_region = ignore_region < 0.5
        ignore_region = torch.from_numpy(ignore_region.copy()).unsqueeze(1).bool()

        max_trimap_kernel_size = 13
        eps = rnd.uniform(0.01,0.2)
        tri, a = make_trimap(rnd, a, eps=eps, dilation_kernel=rnd.randint(0,max_trimap_kernel_size), close_first=rnd.uniform(0,1)<0.05, ignore_region=ignore_region)
        
        return fg, bg, a, 0, tri, torch.tensor(idx)


class DIM_Train(torch.utils.data.Dataset):
    def __init__(self, data_root, image_shape,
                 mode='train',
                 sample_length=3,
                 do_histogram_matching=0.5,
                 do_gamma_aug=0.5,
                 do_jpeg_aug=0.5,
                 do_gaussian_aug=0.5,
                 do_motion_aug=0.5,):
        self.mode = mode
        self.sample_length = sample_length
        assert self.mode in ['train', 'val']
        self.data_root = data_root
        self.image_shape = list(image_shape)
        self.do_histogram_matching = do_histogram_matching
        self.do_gamma_aug = do_gamma_aug
        self.do_jpeg_aug = do_jpeg_aug
        self.do_gaussian_aug = do_gaussian_aug
        self.do_motion_aug = do_motion_aug

        self.FG = list()
        self.BG = list()
        self.Alpha = list()
        
        self.data_root_DIM = os.path.join(self.data_root, 'Combined_Dataset')
        DIM_train, DIM_test = self.parse_DIM(self.data_root_DIM)
        self.FG += DIM_train['fg']
        self.BG += DIM_train['bg']
        
        self.FG_len = len(self.FG)
        self.BG_len = len(self.BG)
        
        self.pixel_aug_gamma = iaa.GammaContrast(gamma=iap.TruncatedNormal(1.0, 0.2, 0.5, 1.5))
        self.pixel_aug_gaussian = iaa.AdditiveGaussianNoise(scale=(0, 0.03*255))
        self.jpeg_aug = iaa.JpegCompression(compression=(20, 80)) 
        self.motion_aug = A.MotionBlur(p=1.0, blur_limit=(3,50))


    def __len__(self):
        return self.FG_len

    def parse_DIM(self, data_root):
        if os.path.exists(os.path.join(data_root, 'mymeta.pkl')):
            with open(os.path.join(data_root, 'mymeta.pkl'), 'rb') as f:
                data = pickle.load(f)

        else:
            modes = ['train', 'test']
            path_txt = dict()
            path_txt['train'] = dict()
            path_txt['train']['fg'] = 'Training_set/training_fg_names.txt'
            path_txt['train']['bg'] = 'Training_set/training_bg_names.txt'
            path_txt['train']['tri'] = None
            path_txt['test'] = dict()
            path_txt['test']['fg'] = 'Test_set/test_fg_names.txt'
            path_txt['test']['bg'] = 'Test_set/test_bg_names.txt'
            path_txt['test']['tri'] = None
            path = dict()
            path['train'] = dict()
            path['train']['fg'] = ['Training_set/Adobe-licensed images', 'Training_set/Other']
            path['train']['bg'] = ['Training_set/train2014']
            path['train']['tri'] = None
            path['test'] = dict()
            path['test']['fg'] = ['Test_set/Adobe-licensed images']
            path['test']['bg'] = ['Test_set/VOCdevkit/VOC2008/JPEGImages']
            path['test']['tri'] = ['Test_set/Adobe-licensed images/trimaps']
            data = dict()
            print('making meta for DIM dataset')
            for mode in modes:
                data[mode] = dict()
                for fbg in ['fg', 'bg', 'tri']:
                    if path[mode][fbg] is None:
                        data[mode][fbg] = None
                    else:
                        print('%s - %s'%(mode, fbg))
                        data_current = list()

                        add_path = '/fg' if fbg == 'fg' else ''
                        file_lists = list()
                        dir_lists = list()
                        for dir_name in sorted(path[mode][fbg]):
                            for file_name in sorted(os.listdir(os.path.join(data_root, dir_name+add_path))):
                                file_lists.append(file_name)
                                dir_lists.append(dir_name)

                        if path_txt[mode][fbg] is not None:
                            FBG_list = list()
                            with open(os.path.join(data_root, path_txt[mode][fbg]), 'r') as f:
                                for v in f:
                                    v = v.strip()
                                    FBG_list.append(v)
                            for img_name in tqdm.tqdm(FBG_list):
                                idx = file_lists.index(img_name)
                                if fbg == 'fg':
                                    data_current.append(['DIM',
                                                         [os.path.join(dir_lists[idx], 'fg', img_name),
                                                         os.path.join(dir_lists[idx], 'alpha', img_name)]])
                                else:
                                    data_current.append(['DIM',os.path.join(dir_lists[idx], img_name)])
                        else:
                            for idx, img_name in enumerate(tqdm.tqdm(file_lists)):
                                data_current.append(['DIM', os.path.join(dir_lists[idx], img_name)])
                        
                        data[mode][fbg] = data_current

            with open(os.path.join(data_root, 'mymeta.pkl'), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
        return data['train'], data['test']

    def random_crop(self, frame, mask, size, rnd):
        do_mask = mask is not None
        ## random transformations that both to be applied.
        min_scale = np.maximum(size[0]/np.float(frame.shape[0]), size[1]/np.float(frame.shape[1]))

        for t in range(10):
            scale = np.maximum(rnd.choice([1., 1./1.5, 1./2.]), min_scale+0.01)
            dsize = (np.int(frame.shape[1]*scale), np.int(frame.shape[0]*scale))
            trans_frame  = cv2.resize(frame, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            if do_mask:
                trans_mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            
            ## try to crop patch that contains object area if cant just return
            np_in1 = None
            for tt in range(1000):
                cr_y = rnd.randint(0, trans_frame.shape[0] - size[0])
                cr_x = rnd.randint(0, trans_frame.shape[1] - size[1])
                if do_mask:
                    center_alpha_val = trans_mask[cr_y+int(size[0]//2), cr_x+int(size[1]//2)]
                    if (tt < 900) or (t < 5):
                        if (center_alpha_val > (0.2*255)) and (center_alpha_val < (0.8*255)):
                            crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
                            break
                    else:
                        if np_in1 is None:
                            np_in1 = np.sum((trans_mask > (0.2*255)) & (trans_mask < (0.8*255)))
                        crop_mask = trans_mask[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
                        crop_mask_UR = (crop_mask > (0.2*255)) & (crop_mask < (0.8*255))
                        if (np.sum(crop_mask_UR) > 0.5*np_in1) or np.mean(crop_mask_UR) > 0.01/255.:
                            break
                else:
                    crop_mask = None
                    break

            if tt < 999:
                break 
        crop_frame = trans_frame[cr_y:cr_y+size[0], cr_x:cr_x+size[1]]
        
        return crop_frame, crop_mask


    def __getitem__(self, idx):
        rnd = random.Random()
        data_FG, sample_FG = self.FG[idx]
        idx_BG = rnd.randint(0, self.BG_len-1)
        data_BG, sample_BG = self.BG[idx_BG]

        # img I/O
        fg = cv2.imread(os.path.join(self.data_root_DIM, sample_FG[0]), cv2.IMREAD_COLOR)
        alpha = cv2.imread(os.path.join(self.data_root_DIM, sample_FG[1]), cv2.IMREAD_GRAYSCALE)
        bg = cv2.imread(os.path.join(self.data_root_DIM, sample_BG), cv2.IMREAD_COLOR)

        fg_list, bg_list, a_list = [None] * self.sample_length, [None] * self.sample_length, [None] * self.sample_length
        for i in range(self.sample_length):
            _fg, _alpha = self.random_crop(fg.copy(), alpha.copy(), self.image_shape, rnd)
            _bg, _ = self.random_crop(bg.copy(), None, self.image_shape, rnd)
            fg_list[i] = _fg
            bg_list[i] = _bg
            a_list[i] = _alpha

        fg = np.stack(fg_list, axis=0)
        bg = np.stack(bg_list, axis=0)
        a = np.stack(a_list, axis=0)

        # gamma augmentation
        if (rnd.uniform(0,1) < self.do_gamma_aug):
            fg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(fg_aug.augment_image(np.uint8(fg[i])))
        
        if (rnd.uniform(0,1) < self.do_gamma_aug) and (bg[0] is not None):
            bg_aug = self.pixel_aug_gamma.to_deterministic()
            for i in range(self.sample_length):
                bg[i] = np.float32(bg_aug.augment_image(np.uint8(bg[i])))

        if (rnd.uniform(0,1) < self.do_histogram_matching):
            ratio = rnd.uniform(0,0.5)
            if rnd.uniform(0,1) < 0.05:
                bg_match = exposure.match_histograms(bg, fg, channel_axis=-1)
                bg = bg_match * ratio + bg * (1. - ratio)
            else:
                fg_match = exposure.match_histograms(fg, bg, channel_axis=-1)
                fg = fg_match * ratio + fg * (1. - ratio)

        # flip all
        if rnd.uniform(0,1) <  0.5:
            fg = _flip_axis(fg, 2)
            a = _flip_axis(a, 2)
        if rnd.uniform(0,1) <  0.5:
            bg = _flip_axis(bg, 2)

        for i in range(self.sample_length):
            # flip one
            if rnd.uniform(0,1) <  0.05:
                fg[i] = _flip_axis(fg[i], 1)
                bg[i] = _flip_axis(bg[i], 1)
                a[i] = _flip_axis(a[i], 1)

        # motion augmentation
        if (rnd.uniform(0,1) < self.do_motion_aug):
            if rnd.uniform(0,1) < 0.5:
                N_cat = np.concatenate([fg, bg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                fg = N_cat_aug[..., :3]
                bg = N_cat_aug[..., 3:6]
                a = N_cat_aug[..., 6]
                fg = np.clip(fg, 0, 255)
                bg = np.clip(bg, 0, 255)
                a = np.clip(a, 0, 255)
            else:
                if rnd.uniform(0,1) < 0.9:
                    N_cat = np.concatenate([fg, a[:,:,:,np.newaxis]], axis=3) # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    fg = N_cat_aug[..., :3]
                    a = N_cat_aug[..., 3]
                    fg = np.clip(fg, 0, 255)
                    a = np.clip(a, 0, 255)
                if rnd.uniform(0,1) < 0.3:
                    N_cat = bg # t,h,w,7
                    N_cat = N_cat.transpose((1,2,3,0)) # h,w,7,t
                    N_cat = N_cat.reshape(self.image_shape[0], self.image_shape[1], -1) # h,w,7*t
                    N_cat_aug = self.motion_aug(image=N_cat)["image"] # h,w,7*t
                    N_cat_aug = N_cat_aug.reshape(self.image_shape[0], self.image_shape[1], -1, self.sample_length) # h,w,7,t
                    N_cat_aug = N_cat_aug.transpose((3,0,1,2)) # t,h,w,7
                    bg = N_cat_aug
                    bg = np.clip(bg, 0, 255)

        # augmentation
        if (rnd.uniform(0,1) < self.do_gaussian_aug):
            aug = self.pixel_aug_gaussian.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))
        if (rnd.uniform(0,1) < self.do_jpeg_aug):
            aug = self.jpeg_aug.to_deterministic()
            for i in range(self.sample_length):
                fg[i] = np.float32(aug.augment_image(np.uint8(fg[i])))
                bg[i] = np.float32(aug.augment_image(np.uint8(bg[i])))
                a[i] = np.float32(aug.augment_image(np.uint8(a[i])))

        # random affine
        ignore_region = np.ones_like(a)
        for i in range(self.sample_length):
            fg[i], bg[i], a[i], ignore_region[i] = random_transform([fg[i], bg[i], a[i], ignore_region[i]], rnd, rt=25, sh=15, zm=[0.90,1.10], sc= [0.9, 1.0], cs=0.07*255, hf=False)
    
        a = a / 255.

        fg = torch.from_numpy(np.transpose(fg, (0, 3, 1, 2)).copy()).float()
        bg = torch.from_numpy(np.transpose(bg, (0, 3, 1, 2)).copy()).float()
        a = torch.from_numpy(a.copy()).unsqueeze(1).float()

        ignore_region = ignore_region < 0.5
        
        ignore_region = torch.from_numpy(ignore_region.copy()).unsqueeze(1).bool()
        max_trimap_kernel_size = 13
        eps = rnd.uniform(0.01,0.2)
        close_first = rnd.uniform(0,1)<0.05
        tri, a = make_trimap(rnd, a, eps=eps, dilation_kernel=rnd.randint(0,max_trimap_kernel_size), close_first=close_first, ignore_region=ignore_region)
        
        return fg, bg, a, 0, tri, torch.tensor(idx)


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self,data_name, data_root, FG, BG, a, tri_gt,
                 trimap=None,
                 num_frames=1,
                 max_image_shape=(1080,1920)):
        self.data_name = data_name
        self.data_root = data_root
        self.FG = FG
        self.BG = BG
        self.a = a
        self.tri_gt = tri_gt
        self.trimap = trimap
        self.num_frames = num_frames
        self.max_image_shape = max_image_shape
        
        self.dataset_length = len(self.FG)

        if type(self.data_root) == list:
            self.data_root_FG = self.data_root[0]
            self.data_root_BG = self.data_root[1]
        else:
            self.data_root_FG = self.data_root
            self.data_root_BG = self.data_root

        if self.data_name == 'DVM2':
            self.max_image_shape = (9999, 9999)
        
        self.eps = [0., 1.]
        
    def __len__(self):
        return self.dataset_length

    def img_crop_and_resize(self, img, ph, pw, nsize=None, mode='bilinear'):
        img2 = img[ph:ph+nsize[0], pw:pw+nsize[1]] if nsize is not None else img
        img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
        return img2
        
    def get_data(self, idx):
        fgp = os.path.join(self.data_root_FG, self.FG[idx])
        
        _f = cv2.imread(fgp, cv2.IMREAD_UNCHANGED)
        if self.a is None:
            if self.tri_gt is None:
                fg = np.float32(_f[..., :-1])
                a = np.float32(_f[..., -1:]) / 255.

            else:
                fg = _f
                a = np.float32(np.ones_like(fg[..., -1:]))
        
        if fg.shape[-1] > 3:
            fg = fg[..., :3]
            

        a[a < self.eps[0]] = 0.
        a[a > self.eps[1]] = 1.
        
        if self.tri_gt is None:
            tri_gt = 0
        else:
            _tri_gt = cv2.imread(os.path.join(self.data_root_FG, self.tri_gt[idx]), cv2.IMREAD_UNCHANGED)
            _tri_gt = _tri_gt > 1
            tri_gt = np.float32(np.zeros_like(_tri_gt))
            tri_gt[..., 0][np.logical_not(_tri_gt[..., 1] + _tri_gt[..., 2])] = 1
            tri_gt[..., 1][_tri_gt[..., 2]] = 1
            tri_gt[..., 2][_tri_gt[..., 1]] = 1


        if self.BG is None:
            bg = fg
        else:
            bgp = os.path.join(self.data_root_BG, self.BG[idx])
            if not os.path.exists(bgp):
                bgp = os.path.splitext(bgp)[0]+'.png'
            bg = np.float32(cv2.imread(bgp, cv2.IMREAD_COLOR))
        
        fg = self.img_crop_and_resize(fg, 0, 0).float()
        if self.tri_gt is not None:
            tri_gt = self.img_crop_and_resize(tri_gt, 0, 0).float()
        bg = self.img_crop_and_resize(bg, 0, 0).float()
        a = self.img_crop_and_resize(a, 0, 0).float()

        filename = os.path.splitext(os.path.basename(self.FG[idx]))[0]+'.jpg'

        if self.trimap is None:
            tri = 0
        else:
            tri_path = os.path.join(self.trimap, filename)
            if not os.path.exists(tri_path):
                raise FileNotFoundError('Cannot find trimap image files: {}'.format(tri_path))
            tri = np.float32(cv2.imread(tri_path, cv2.IMREAD_COLOR))
            tri = self.img_crop_and_resize(tri, 0, 0).float()

        return fg, bg, a, self.eps, tri_gt, tri, torch.tensor(idx), filename

    def __getitem__(self, idx):
        if self.num_frames == 1:
            return self.get_data(idx)
        elif self.num_frames > 1:
            frame_idx = (np.arange(idx - (self.num_frames/2), idx + (self.num_frames/2), 1) + 0.5).astype(np.int32)
            frame_idx = np.clip(frame_idx, a_min=0, a_max=self.__len__()-1)

            fg = list()
            bg = list()
            a = list()
            tri_gt = list()
            tri = list()
            filenames = list()
            for _idx in frame_idx:
                _fg, _bg, _a, _, _tri_gt, _tri, _, _filename = self.get_data(_idx)
                fg.append(_fg)
                bg.append(_bg)
                a.append(_a)
                tri_gt.append(_tri_gt)
                tri.append(_tri)
                filenames.append(_filename)
            
            fg = torch.cat(fg, dim=0)
            bg = torch.cat(bg, dim=0)
            a = torch.cat(a, dim=0)
            if self.tri_gt is None:
                tri_gt = 0
            else:
                tri_gt = torch.cat(tri_gt, dim=0)
            if self.trimap is None:
                tri = 0
            else:
                tri = torch.cat(tri, dim=0)
            filename = filenames[int(self.num_frames/2)]

            return fg, bg, a, self.eps, tri_gt, tri, torch.tensor(idx), filename

class VideoMatting108_Test():
    FG_FOLDER = 'FG_done'
    BG_FOLDER = 'BG_done2'
    FLOW_FOLDER = 'flow_png'
    def __init__(self, data_root,
                 mode='val',
                 use_subset=False,
                 ):
        self.idx = 0
        self.mode = mode
        assert self.mode in ['train', 'val']
        self.data_root_V108 = os.path.join(data_root, 'VideoMatting108')
        setname = '{}_videos_subset.txt' if use_subset else '{}_videos.txt'
        setname = setname.format(self.mode)

        with open(os.path.join(self.data_root_V108, 'frame_corr.json'), 'r') as f:
            self.frame_corr = json.load(f)
        with open(os.path.join(self.data_root_V108, setname), 'r') as f:
            self.FG, self.BG, self.seq_name = self.parse_VideoMatting108(f, self.frame_corr, self.data_root_V108, self.FG_FOLDER, self.BG_FOLDER)
        
        self.FG_len = len(self.FG)
        self.BG_len = len(self.BG)

    def __len__(self):
        return self.FG_len

    def parse_VideoMatting108(self, f, frame_corr, data_root, FG_FOLDER, BG_FOLDER):
        FG = list()
        BG = list()
        seq_name = list()
        for v in f:
            FG_path_current = list()
            BG_path_current = list()
            v = v.strip()
            fns = [k for k in sorted(self.frame_corr.keys()) if os.path.dirname(k) == v]
            for i in range(len(fns)):
                FG_path_current.append(os.path.join(FG_FOLDER, fns[i]))
                BG_path_current.append(os.path.join(BG_FOLDER, frame_corr[fns[i]]))
            FG.append(FG_path_current)
            BG.append(BG_path_current)
            seq_name.append(v)
        return FG, BG, seq_name

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx
        if self.idx >= len(self):
            raise StopIteration
        
        fg = self.FG[idx]
        bg = self.BG[idx]
        a = None
        tri = None
        seq_name = self.seq_name[idx]

        self.idx += 1
        return 'V108', self.data_root_V108, fg, bg, a, tri, seq_name
