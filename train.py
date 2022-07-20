import argparse
from io import UnsupportedOperation
import logging
import os
import shutil
import time
import timeit
import shutil

import numpy as np
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torchvision.utils import save_image

from config import get_cfg_defaults
from dataset import DIM_Train, VideoMatting108_Train
from helpers import *
from utils.optimizer import RAdam

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--gpu", type=str, default='0,1,2,3')
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.TRAIN.STAGE = args.stage
    cfg.freeze()

    return args, cfg

def main(args, cfg):
    MODEL = get_model_name(cfg)
    random_seed = cfg.SYSTEM.RANDOM_SEED
    base_lr = cfg.TRAIN.BASE_LR
    
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    output_dir = os.path.join(cfg.SYSTEM.OUTDIR, 'checkpoint')
    if args.local_rank <= 0:
        os.makedirs(output_dir, exist_ok=True)
    start = timeit.default_timer()
    # cudnn related setting
    cudnn.benchmark = cfg.SYSTEM.CUDNN_BENCHMARK
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        if args.local_rank <= 0:
            print('Seeding with', random_seed)
        random.seed(random_seed+args.local_rank)
        torch.manual_seed(random_seed+args.local_rank)

    args.world_size = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.local_rank >= 0:
        device = torch.device('cuda:{}'.format(args.local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
        args.world_size = torch.distributed.get_world_size()
    else:
        if torch.cuda.is_available():
            print('using Cuda devices, num:', torch.cuda.device_count())

    if args.local_rank <= 0:
        logger, final_output_dir = create_logger(output_dir, MODEL, 'train')
        print(cfg)
        with open(os.path.join(final_output_dir, 'config.yaml'), 'w') as f:
            f.write(str(cfg))
        image_outdir = os.path.join(final_output_dir, 'training_images')
        os.makedirs(os.path.join(final_output_dir, 'training_images'), exist_ok=True)
    else:
        image_outdir = None

    if cfg.TRAIN.STAGE == 1:
        model_trimap = None
    else:
        model_trimap = get_model_trimap(cfg, mode='Train')
    model = get_model_alpha(cfg, model_trimap, mode='Train')


    if cfg.TRAIN.STAGE == 1:
        load_ckpt = './weights/FBA.pth'
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in dct.keys():
            dct = dct['state_dict']
        missing_keys, unexpected_keys = model.NET.load_state_dict(dct, strict=False)
        if args.local_rank <= 0:
            logger.info('Missing keys: ' + str(sorted(missing_keys)))
            logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
            logger.info("=> loaded checkpoint from Image Matting Weight: {}".format(load_ckpt))
    elif cfg.TRAIN.STAGE in [2,3]:
        load_ckpt = './weights/s1_OTVM_trimap.pth'
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = model.trimap.model.load_state_dict(dct, strict=False)
        if args.local_rank <= 0:
            logger.info('Missing keys: ' + str(sorted(missing_keys)))
            logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
            logger.info("=> loaded checkpoint from Pretrained STM Weight: {}".format(load_ckpt))
        
        if cfg.TRAIN.STAGE == 2:
            load_ckpt = './weights/s1_OTVM_alpha.pth'
        elif cfg.TRAIN.STAGE == 3:
            load_ckpt = './weights/s2_OTVM_alpha.pth'
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = model.NET.load_state_dict(dct, strict=False)
        if args.local_rank <= 0:
            logger.info('Missing keys: ' + str(sorted(missing_keys)))
            logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
    elif cfg.TRAIN.STAGE == 4:
        load_ckpt = './weights/s3_OTVM.pth'
        dct = torch.load(load_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(dct)

    torch_barrier()

    ADDITIONAL_INPUTS = dict()
    
    start_epoch = 0

    if args.local_rank >= 0:
        # FBA particularly uses batch_size == 1, thus no syncbn here
        if (not cfg.ALPHA.MODEL.endswith('fba')) and (not cfg.TRAIN.FREEZE_BN):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        find_unused_parameters = False
        if cfg.TRAIN.STAGE == 2:
            find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=find_unused_parameters,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    else:
        model = torch.nn.DataParallel(model).cuda()

    if cfg.TRAIN.STAGE in [2,3]:
        params = list()
        for k, v in model.named_parameters():
            if v.requires_grad:
                _k = k[7:] # remove 'module.'
                if _k.startswith('NET.'):
                    if cfg.TRAIN.STAGE == 3:
                        if args.local_rank <= 0:
                            logging.info('do NOT train parameter: %s'%(k))
                        pass
                    else:
                        params.append({'params': v, 'lr': base_lr})
                elif _k.startswith('trimap.'):
                    if cfg.TRAIN.STAGE == 2:
                        if args.local_rank <= 0:
                            logging.info('do NOT train parameter: %s'%(k))
                        pass
                    else:
                        params.append({'params': v, 'lr': base_lr})
                else:
                    if args.local_rank <= 0:
                        logging.info('%s: Undefined parameter'%(k))
                    params.append({'params': v, 'lr': base_lr})
    else:
        params_dict = {k: v for k, v in model.named_parameters() if v.requires_grad}
        params = [{'params': list(params_dict.values()), 'lr': base_lr}]
        
    params_count = 0
    if args.local_rank <= 0:
        logging.info('=> Parameters needs to be optimized:')
        for param in params:
            _param = param['params']
            if type(_param) is list:
                for _p in _param:
                    params_count += _p.shape.numel()
            else:
                params_count += _param.shape.numel()
        logging.info('=> Total Parameters: {}'.format(params_count))
        
    
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(params, lr=base_lr)
    elif cfg.TRAIN.OPTIMIZER == 'radam':
        optimizer = RAdam(params, lr=base_lr, weight_decay=weight_decay)
    
    if cfg.TRAIN.LR_STRATEGY == 'stair':
        adjust_lr = stair_lr
    elif cfg.TRAIN.LR_STRATEGY == 'poly':
        adjust_lr = poly_lr
    elif cfg.TRAIN.LR_STRATEGY == 'const':
        adjust_lr = const_lr
    else:
        raise NotImplementedError('[%s] is not supported in cfg.TRAIN.LR_STRATEGY'%(cfg.TRAIN.LR_STRATEGY))

    total_epochs = cfg.TRAIN.TOTAL_EPOCHS

    sample_length = cfg.TRAIN.FRAME_NUM
    if cfg.TRAIN.STAGE == 1:
        sample_length = 1
    if cfg.TRAIN.STAGE in [1,2,3]:
        train_dataset = DIM_Train(
                            data_root=cfg.DATASET.PATH,
                            image_shape=cfg.TRAIN.TRAIN_INPUT_SIZE,
                            mode='train',
                            sample_length=sample_length,
                        )
    else:
        train_dataset = VideoMatting108_Train(
            data_root=cfg.DATASET.PATH,
            image_shape=cfg.TRAIN.TRAIN_INPUT_SIZE,
            mode='train',
            sample_length=sample_length,
            max_skip=15,
            do_affine=0.5,
            do_time_flip=0.5,
        )

    if cfg.SYSTEM.TESTMODE:
        start_epoch = max(start_epoch, total_epochs - 1)
    for epoch in range(start_epoch, total_epochs):
        train(epoch, cfg, args, train_dataset, base_lr, start_epoch, total_epochs,
              optimizer, model, adjust_lr, image_outdir, MODEL,
              ADDITIONAL_INPUTS)
        if args.local_rank <= 0:
            if (((epoch+1) % cfg.TRAIN.SAVE_EVERY_EPOCH) == 0) or ((epoch+1) == total_epochs):
                weight_fn = os.path.join(final_output_dir, 'checkpoint_{}.pth'.format(epoch+1))
                logger.info('=> saving checkpoint to {}'.format(weight_fn))
                if cfg.TRAIN.STAGE in [1,2]:
                    torch.save(model.module.NET.state_dict(), weight_fn)
                else:
                    torch.save(model.module.state_dict(), weight_fn)
                optim_fn = os.path.join(final_output_dir, 'optim_{}.pth'.format(epoch+1))
                torch.save(optimizer.state_dict(), optim_fn)
        
    if args.local_rank <= 0:
        weight_fn = os.path.join('weights', '{:s}.pth'.format(MODEL))
        logger.info('=> saving checkpoint to {}'.format(weight_fn))
        if cfg.TRAIN.STAGE in [1,2]:
            torch.save(model.module.NET.state_dict(), weight_fn)
        else:
            torch.save(model.module.state_dict(), weight_fn)

    end = timeit.default_timer()
    if args.local_rank <= 0:
        logger.info('Time: %d sec.' % np.int32((end-start)))
        logger.info('Done')



def write_image(outdir, out, step, max_batch=1, trimap=False):
    with torch.no_grad():
        scaled_imgs, scaled_tris, alphas, comps, gts, fgs, bgs = out[:7]
        if trimap:
            pred_tris = out[7]
        b, s, _, h, w = scaled_imgs.shape
        b = max_batch if b > max_batch else b
        img_list = list()
        img_list.append(scaled_imgs[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(scaled_tris[:max_batch].reshape(b*s, 1, h, w).expand(-1, 3, -1, -1))
        img_list.append(gts[:max_batch].reshape(b*s, 1, h, w).expand(-1, 3, -1, -1))
        img_list.append(alphas[:max_batch].reshape(b*s, 1, h, w).expand(-1, 3, -1, -1))
        if trimap:
            img_list.append(pred_tris[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(comps[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(fgs[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(bgs[:max_batch].reshape(b*s, 3, h, w))
        imgs = torch.cat(img_list, dim=0).reshape(-1, 3, h, w)
        if h > 320:
            imgs = F.interpolate(imgs, scale_factor=320/h)
        save_image(imgs, os.path.join(outdir, '{}.png'.format(step)), nrow=int(s*b))

def train(epoch, cfg, args, train_dataset, base_lr, start_epoch, total_epochs,
          optimizer, model, adjust_learning_rate, image_outdir, MODEL,
          ADDITIONAL_INPUTS):    
    # Training
    torch.cuda.empty_cache()
    if cfg.TRAIN.STAGE in [1,2,3]:
        train_dataset_concat = [train_dataset] * 20
    else:
        if epoch < 100:
            SKIP = min(1+(epoch//5), 25)
        else:
            SKIP = max(44-(epoch//5), 10)
        train_dataset.max_skip = SKIP
        train_dataset_concat = [train_dataset] * 20
    
    train_dataset = data.ConcatDataset(train_dataset_concat)
    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE // args.world_size),
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        shuffle=True if train_sampler is None else False,
        sampler=train_sampler)
    
    if args.local_rank >= 0:
        train_sampler.set_epoch(epoch)

    iters_per_epoch = len(trainloader)
    image_freq = cfg.TRAIN.IMAGE_FREQ if cfg.TRAIN.IMAGE_FREQ > 0 else 1e+8
    image_freq = min(image_freq, iters_per_epoch)
   
    # STM DISABLES BN DURING TRAINING
    model.train()
    if cfg.TRAIN.STAGE > 1:
        for m in model.module.trimap.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() # turn-off BN
    if cfg.TRAIN.FREEZE_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval() # turn-off BN
    if cfg.TRAIN.STAGE == 2:
        model.module.trimap.eval()
        if args.local_rank <= 0:
            logging.info('Set trimap model to eval mode')
    if cfg.TRAIN.STAGE == 3:
        model.module.NET.eval()
        if args.local_rank <= 0:
            logging.info('Set alpha model to eval mode')
    
    sub_losses = ['L_alpha', 'L_comp', 'L_grad'] if not cfg.ALPHA.MODEL.endswith('fba') else \
                ['L_alpha_comp', 'L_lap', 'L_grad']
    
    data_time = AverageMeter()
    losses = AverageMeter()
    sub_losses_avg = [AverageMeter() for _ in range(len(sub_losses))]
    tic = time.time()
    cur_iters = epoch*iters_per_epoch

    prefetcher = data_prefetcher(trainloader)
    dp = prefetcher.next()
    i_iter = 0
    while dp[0] is not None:
        if cfg.SYSTEM.TESTMODE:
            if i_iter > 20:
                print()
                break
        def step(i_iter, dp, tic):
            data_time.update(time.time() - tic)
            
            def handle_batch():
                fg, bg, a, ir, tri, _ = dp      # [B, 3, 3 or 1, H, W]

                bg = bg if bg.dim() > 1 else None
                a = a if a.dim() > 1 else None
                ir = ir if ir.dim() > 1 else None

                out = model(a, fg, bg, ignore_region=ir, tri=tri)
                L_alpha = out[0].mean()
                L_comp = out[1].mean()
                L_grad = out[2].mean()
                vis_alpha = L_alpha.detach()#.item()
                vis_comp = L_comp.detach()#.item()
                vis_grad = L_grad.detach()#.item()
                if cfg.TRAIN.STAGE == 1:
                    loss = L_alpha + L_comp + L_grad
                    batch_out = [loss.detach(), vis_alpha, vis_comp, vis_grad, out[4:-1]]
                else:
                    L_tri = out[3].mean()
                    loss = L_alpha + L_comp + L_grad + L_tri
                    batch_out = [loss.detach(), vis_alpha, vis_comp, vis_grad, out[4:]]

                model.zero_grad()
                loss.backward()
                optimizer.step()

                return batch_out

            loss, vis_alpha, vis_comp, vis_grad, vis_images = handle_batch()

            reduced_loss = reduce_tensor(loss)
            reduced_sub_losses = [reduce_tensor(vis_alpha), reduce_tensor(vis_comp), reduce_tensor(vis_grad)]

            # update average loss
            losses.update(reduced_loss.item())
            sub_losses_avg[0].update(reduced_sub_losses[0].item())
            sub_losses_avg[1].update(reduced_sub_losses[1].item())
            sub_losses_avg[2].update(reduced_sub_losses[2].item())

            torch_barrier()

            current_lr = adjust_learning_rate(optimizer,
                                base_lr,
                                total_epochs * iters_per_epoch,
                                i_iter+cur_iters)

            if args.local_rank <= 0:
                progress_bar(i_iter, iters_per_epoch, epoch, start_epoch, total_epochs, 'finetuning',
                                'Data: {data_time} | '
                                'Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                                '{sub_losses[0]}: {sub_losses_avg[0].val:.4f} ({sub_losses_avg[0].avg:.4f})'.format(
                                data_time=format_time(data_time.sum),
                                loss=losses,
                                sub_losses=sub_losses,
                                sub_losses_avg=sub_losses_avg))
            
            if i_iter % image_freq == 0 and args.local_rank <= 0:
                write_image(image_outdir, vis_images, i_iter+cur_iters, trimap=(cfg.TRAIN.STAGE > 1))
            return current_lr
        
        current_lr = step(i_iter, dp, tic)
        tic = time.time()
        
        dp = prefetcher.next()
        i_iter += 1

    if args.local_rank <= 0:
        logger_str = '{:s} | E [{:d}] | I [{:d}] | LR [{:.1e}] | Total Loss:{: 4.6f}'.format(
                    MODEL, epoch+1, i_iter+1, current_lr, losses.avg)
        logger_str += ' | {} [{: 4.6f}] | {} [{: 4.6f}] | {} [{: 4.6f}]'.format(
                    sub_losses[0], sub_losses_avg[0].avg, 
                    sub_losses[1], sub_losses_avg[1].avg,
                    sub_losses[2], sub_losses_avg[2].avg)
        logging.info(logger_str)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_fg, self.next_bg, self.next_a, self.next_ir, self.next_tri, self.next_idx = next(self.loader)
        except StopIteration:
            self.next_fg = None
            self.next_bg = None
            self.next_a = None
            self.next_ir = None
            self.next_tri = None
            self.next_idx = None
            return
        with torch.cuda.stream(self.stream):
            self.next_fg = self.next_fg.cuda(non_blocking=True)
            self.next_bg = self.next_bg.cuda(non_blocking=True)
            self.next_a = self.next_a.cuda(non_blocking=True)
            self.next_ir = self.next_ir.cuda(non_blocking=True)
            self.next_tri = self.next_tri.cuda(non_blocking=True)
            self.next_idx = self.next_idx.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        fg = self.next_fg
        bg = self.next_bg
        a = self.next_a
        ir = self.next_ir
        tri = self.next_tri
        idx = self.next_idx
        if fg is not None:
            fg.record_stream(torch.cuda.current_stream())
        if bg is not None:
            bg.record_stream(torch.cuda.current_stream())
        if a is not None:
            a.record_stream(torch.cuda.current_stream())
        if ir is not None:
            ir.record_stream(torch.cuda.current_stream())
        if tri is not None:
            tri.record_stream(torch.cuda.current_stream())
        if idx is not None:
            idx.record_stream(torch.cuda.current_stream())
        self.preload()
        return fg, bg, a, ir, tri, idx




def get_sampler(dataset, shuffle=True):
    if torch_dist.is_available() and torch_dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset, shuffle=shuffle)
    else:
        return None


def IoU(pred, true):
    _, _, n_class, _, _ = pred.shape

    _, xx = torch.max(pred, dim=2)
    _, yy = torch.max(true, dim=2)
    iou = list()
    for n in range(n_class):
        x = (xx == n).float()
        y = (yy == n).float()
        
        i = torch.sum(torch.sum(x*y, dim=-1), dim=-1) # sum over spatial dims
        u = torch.sum(torch.sum((x+y)-(x*y), dim=-1), dim=-1) 

        iou.append(((i + 1e-4) / (u + 1e-4)).mean().item() * 100.) # b
    
    # mean over mini-batch
    return sum(iou)/n_class, iou


if __name__ == "__main__":
    args, cfg = parse_args()
    main(args, cfg)
