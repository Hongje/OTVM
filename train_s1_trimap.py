import argparse
import logging
import os
import time
import timeit
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision.utils import save_image

from config import get_cfg_defaults
from dataset import DIM_Train
from helpers import *
from utils.optimizer import RAdam

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument("--gpu", type=str, default='0,1,2,3')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.freeze()

    return args, cfg


def main(args, cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    MODEL = 's1_OTVM_trimap'
    random_seed = cfg.SYSTEM.RANDOM_SEED
    base_lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    output_dir = os.path.join(cfg.SYSTEM.OUTDIR, 'checkpoint')
    os.makedirs(output_dir, exist_ok=True)
    start = timeit.default_timer()
    # cudnn related setting
    cudnn.benchmark = cfg.SYSTEM.CUDNN_BENCHMARK
    cudnn.deterministic = cfg.SYSTEM.CUDNN_DETERMINISTIC
    cudnn.enabled = cfg.SYSTEM.CUDNN_ENABLED
    if random_seed > 0:
        import random
        print('Seeding with', random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    logger, final_output_dir = create_logger(output_dir, MODEL, 'train')
    print(cfg)
    with open(os.path.join(final_output_dir, 'config.yaml'), 'w') as f:
        f.write(str(cfg))
    image_outdir = os.path.join(final_output_dir, 'training_images')
    os.makedirs(os.path.join(final_output_dir, 'training_images'), exist_ok=True)

    model = get_model_trimap(cfg, mode='Train')
    torch_barrier()

    start_epoch = 0

    load_ckpt = './weights/STM_weights.pth'
    dct = load_NoPrefix(load_ckpt, 7)
    missing_keys, unexpected_keys = model.model.load_state_dict(dct, strict=False)
    logger.info('Missing keys: ' + str(sorted(missing_keys)))
    logger.info('Unexpected keys: ' + str(sorted(unexpected_keys)))
    logger.info("=> loaded checkpoint from {}".format(load_ckpt))

    model = torch.nn.DataParallel(model).cuda()

    # optimizer
    params_dict = {k: v for k, v in model.named_parameters() if v.requires_grad}
        
    params_count = 0
    logging.info('=> Parameters needs to be optimized:')
    for k in sorted(params_dict):
        params_count += params_dict[k].shape.numel()
    logging.info('=> Total Parameters: {}'.format(params_count))
        
    params = [{'params': list(params_dict.values()), 'lr': base_lr}]
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

    train_dataset = DIM_Train(
                        data_root=cfg.DATASET.PATH,
                        image_shape=cfg.TRAIN.TRAIN_INPUT_SIZE,
                        mode='train',
                        sample_length=3,
                    )
    train_dataset = [train_dataset] * 20

    train_dataset = data.ConcatDataset(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
        shuffle=True)

    if cfg.SYSTEM.TESTMODE:
        start_epoch += 199
    for epoch in range(start_epoch, total_epochs):
        train(epoch, cfg, trainloader, base_lr, start_epoch, total_epochs,
              optimizer, model, adjust_lr, image_outdir, MODEL)

        if (((epoch+1) % cfg.TRAIN.SAVE_EVERY_EPOCH) == 0) or ((epoch+1) == total_epochs):
            weight_fn = os.path.join(final_output_dir, 'checkpoint_{}.pth'.format(epoch+1))
            logger.info('=> saving checkpoint to {}'.format(weight_fn))
            torch.save(model.module.model.state_dict(), weight_fn)
            optim_fn = os.path.join(final_output_dir, 'optim_{}.pth'.format(epoch+1))
            torch.save(optimizer.state_dict(), optim_fn)
    
    weight_fn = os.path.join('weights', '{:s}.pth'.format(MODEL))
    logger.info('=> saving checkpoint to {}'.format(weight_fn))
    torch.save(model.module.model.state_dict(), weight_fn)    
    end = timeit.default_timer()
    logger.info('Time: %d sec.' % np.int32((end-start)))
    logger.info('Done')



def write_image(outdir, out, step, max_batch=1):
    with torch.no_grad():
        scaled_imgs, pred, tris, scaled_gts = out
        b, s, _, h, w = scaled_imgs.shape
        b = max_batch if b > max_batch else b
        img_list = list()
        img_list.append(scaled_imgs[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(tris[:max_batch].reshape(b*s, 3, h, w))
        img_list.append(pred[:max_batch].reshape(b*s, 3, h, w))
        imgs = torch.cat(img_list, dim=0).reshape(-1, 3, h, w)
        if h > 320:
            imgs = F.interpolate(imgs, scale_factor=320/h)
        save_image(imgs, os.path.join(outdir, '{}.png'.format(step)), nrow=int(s*b))

def train(epoch, cfg, trainloader, base_lr, start_epoch, total_epochs,
          optimizer, model, adjust_learning_rate, image_outdir, MODEL):    
    # Training
    iters_per_epoch = len(trainloader)
    image_freq = cfg.TRAIN.IMAGE_FREQ if cfg.TRAIN.IMAGE_FREQ > 0 else 1e+8
    image_freq = min(image_freq, iters_per_epoch)
   
    # STM DISABLES BN DURING TRAINING
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval() # turn-off BN
    
    data_time = AverageMeter()
    losses = AverageMeter()
    IOU = AverageMeter()
    tic = time.time()
    cur_iters = epoch*iters_per_epoch

    prefetcher = data_prefetcher(trainloader)
    dp = prefetcher.next()
    i_iter = 0
    while dp[0] is not None:
        data_time.update(time.time() - tic)
        if cfg.SYSTEM.TESTMODE:
            if i_iter > 20:
                print()
                break
            
        def handle_batch():
            fg, bg, a, ir, tri, _ = dp      # [B, 3, 3 or 1, H, W]

            bg = bg if bg.dim() > 1 else None
            a = a if a.dim() > 1 else None
            ir = ir if ir.dim() > 1 else None

            out = model(a, fg, bg, ignore_region=ir, tri=tri)
            loss = out[0].mean()


            model.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.detach(), out[1:]

        loss, vis_out = handle_batch()

        reduced_loss = reduce_tensor(loss)

        # update average loss
        losses.update(reduced_loss.item())

        tri_pred = vis_out[1]
        tri_gt = vis_out[2]
        mIoU, _ = IoU(tri_pred, tri_gt)
        IOU.update(mIoU)
        torch_barrier()

        current_lr = adjust_learning_rate(optimizer,
                            base_lr,
                            total_epochs * iters_per_epoch,
                            i_iter+cur_iters)

        tic = time.time()
        progress_bar(i_iter, iters_per_epoch, epoch, start_epoch, total_epochs, 'finetuning',
                        'Data: {data_time} | '
                        'Loss: {loss.val:.4f} ({loss.avg:.4f}) | '
                        'IOU: {IOU.val:.4f} ({IOU.avg:.4f})'.format(
                        data_time=format_time(data_time.sum),
                        loss=losses,
                        IOU=IOU))
        
        if i_iter % image_freq == 0:
            write_image(image_outdir, vis_out, i_iter+cur_iters)
        
        dp = prefetcher.next()
        i_iter += 1
    
    logger_str = '{:s} | E [{:d}] | I [{:d}] | LR [{:.1e}] | CE:{: 4.6f} | mIoU:{: 4.6f}'
    logger_format = [MODEL, epoch+1, i_iter+1, current_lr, losses.avg, IOU.avg]
    logging.info(logger_str.format(*logger_format))

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
