import os
import argparse
import time
import timeit
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
import tqdm

from config import get_cfg_defaults
from dataset import EvalDataset, VideoMatting108_Test
from helpers import *

torch.set_grad_enabled(False)

EPS = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument('--trimap', default='medium', choices=['narrow', 'medium', 'wide'])
    parser.add_argument("--viz", action='store_true')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.TRAIN.STAGE = 4

    return args, cfg


def main(cfg, args, GPU):
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    MODEL = get_model_name(cfg)
    random_seed = cfg.SYSTEM.RANDOM_SEED
    output_dir = os.path.join(cfg.SYSTEM.OUTDIR, 'alpha')
    start = timeit.default_timer()
    cudnn.benchmark = False
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        print('Seeding with', random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    outdir_tail = os.path.join(args.trimap, MODEL)
    alpha_outdir = os.path.join(output_dir, 'test', outdir_tail)
    viz_outdir_img = os.path.join(output_dir, 'viz', 'img', outdir_tail)
    viz_outdir_vid = os.path.join(output_dir, 'viz', 'vid', outdir_tail)

    if args.trimap == 'narrow':
        dilate_kernel = 5   # width: 11
    elif args.trimap == 'medium':
        dilate_kernel = 12  # width: 25
    elif args.trimap == 'wide':
        dilate_kernel = 20  # width: 41

    model_trimap = get_model_trimap(cfg, mode='Test', dilate_kernel=dilate_kernel)
    model = get_model_alpha(cfg, model_trimap, mode='Test', dilate_kernel=dilate_kernel)
    
    load_ckpt = os.path.join('weights', '{:s}.pth'.format(MODEL))
    dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(dct)
    model = nn.DataParallel(model.cuda())

    valid_dataset = VideoMatting108_Test(
        data_root=cfg.DATASET.PATH,
        mode='val',
    )
    with torch.no_grad():
        eval(args, cfg, valid_dataset, model, alpha_outdir, viz_outdir_img, viz_outdir_vid, args.viz)
    
    end = timeit.default_timer()
    print('done | Total time: {}'.format(format_time(end-start)))

def write_image(outdir, out, filename, max_batch=4):
    with torch.no_grad():
        scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts, comps = out
        b, s, _, h, w = scaled_imgs.shape
        alphas = alphas.expand(-1,-1,3,-1,-1)
        scaled_gts = scaled_gts.expand(-1,-1,3,-1,-1)

        b = max_batch if b > max_batch else b
        img_list = list()
        img_list.append(scaled_imgs[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(comps[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(tri_gt[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(scaled_gts[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(tri_pred[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(alphas[:max_batch].reshape(b*s, 3, h, w))
        imgs = torch.cat(img_list, dim=0).reshape(-1, 3, h, w)

        imgs = F.interpolate(imgs, size=(h//2, w//2), mode='bilinear', align_corners=False)
        
        save_image(imgs, outdir%(filename), nrow=int(s*b*2))

def eval(args, cfg, valid_dataset, model, alpha_outdir, viz_outdir_img, viz_outdir_vid, VIZ):
    model.eval()

    for i_iter, (data_name, data_root, FG, BG, a, tri, seq_name) in enumerate(valid_dataset):
        if cfg.SYSTEM.TESTMODE:
            if i_iter not in [0, len(valid_dataset)-1]:
                continue
        torch.cuda.empty_cache()
        num_frames = 1
        eval_sequence = EvalDataset(
            data_name=data_name,
            data_root=data_root,
            FG=FG,
            BG=BG,
            a=a,
            tri_gt=tri, # GT trimap
            trimap=None,
            num_frames=num_frames,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_sequence,
            batch_size=1,
            # num_workers=cfg.SYSTEM.NUM_WORKERS,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            sampler=None)

        print('[{}/{}] Set FIXED dilate of unknown region: [{}]'.format(i_iter, len(valid_dataset), args.trimap))

        save_path = os.path.join(alpha_outdir, 'pred', seq_name)
        os.makedirs(save_path, exist_ok=True)
        if VIZ:
            visualization_path_img = os.path.join(viz_outdir_img, 'viz', seq_name)
            visualization_path_vid = os.path.join(viz_outdir_vid, 'viz')
            os.makedirs(visualization_path_img, exist_ok=True)
            os.makedirs(visualization_path_vid, exist_ok=True)

        iterations = tqdm.tqdm(eval_loader)
        for i_seq, dp in enumerate(iterations):
            if cfg.SYSTEM.TESTMODE:
                if i_seq > 10:
                    break

            def handle_batch(dp, first_frame, last_frame, memorize, max_memory_num, large_input):
                fg, bg, a, eps, tri_gt, tri, _, filename = dp      # [B, 3, 3 or 1, H, W]

                if tri.dim() == 1:
                    tri = None
                if tri_gt.dim() == 1:
                    tri_gt = None
                
                out = model(a, fg, bg, tri=tri, tri_gt=tri_gt,
                            first_frame=first_frame,
                            last_frame=last_frame,
                            memorize=memorize,
                            max_memory_num=max_memory_num,
                            large_input=large_input,)
                return out, filename[0]

            first_frame = (i_seq==0)
            last_frame = (i_seq==(len(iterations)-1))
            memorize = False
            MEMORY_SKIP_FRAME = cfg.TEST.MEMORY_SKIP_FRAME
            MEMORY_MAX_NUM = cfg.TEST.MEMORY_MAX_NUM
            large_input = False
            if min(dp[0].shape[-2:]) > 1100:
                MEMORY_SKIP_FRAME = int(MEMORY_SKIP_FRAME * 2)
                MEMORY_MAX_NUM = int(MEMORY_MAX_NUM / 2)
                large_input = True
            if MEMORY_SKIP_FRAME > 2:
                memorize = (i_seq % MEMORY_SKIP_FRAME) == 0
            max_memory_num = MEMORY_MAX_NUM
            
            if first_frame:
                print('[{}/{}] {} | {} | Large input: {}'.format(i_iter, len(valid_dataset), seq_name, dp[0].shape[-2:], large_input))
                
            torch.cuda.synchronize()
            out, filename = handle_batch(dp, first_frame, last_frame, memorize, max_memory_num, large_input,)
            torch.cuda.synchronize()

            scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts = out

            green_bg = torch.zeros_like(scaled_imgs)
            green_bg[:,:,1] = 1.
            comps = scaled_imgs * alphas + green_bg * (1. - alphas)
            
            if VIZ:
                frame_path = os.path.join(visualization_path_img, 'f%d.jpg')
            else:
                frame_path = None
            alpha_pred_img = (alphas*255).byte().cpu().squeeze(0).squeeze(0).squeeze(0).numpy()
            filename_for_save = os.path.splitext(filename)[0]+'.png'

            def write_result_images(alpha_pred_img, path, VIZ, frame_path, vis_out, i_seq):
                if VIZ:
                    write_image(frame_path,
                                vis_out,
                                i_seq)
                cv2.imwrite(path, alpha_pred_img)
            
            write_result_images(alpha_pred_img,
                                os.path.join(save_path, filename_for_save),
                                VIZ,
                                frame_path,
                                # [scaled_imgs, tri_pred, tri_gt, alphas, scaled_gts, comps],
                                [scaled_imgs.cpu(), tri_pred.cpu(), tri_gt.cpu(), alphas.cpu(), scaled_gts.cpu(), comps.cpu()],
                                i_seq)


            torch.cuda.synchronize()
        
        if VIZ:
            if '/' in seq_name:
                vid_name = seq_name.split('/')
                vid_name = '_'.join(vid_name)
            else:
                vid_name = seq_name
            vid_path = os.path.join(visualization_path_vid, '{}.mp4'.format(vid_name))

            def make_viz_video(frame_path, vid_path):
                os.system('ffmpeg -framerate 10 -i {} {}  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
                time.sleep(10) # wait 10 seconds

            make_viz_video(frame_path, vid_path)

if __name__ == "__main__":
    args, cfg = parse_args()
    main(cfg, args, args.gpu)
