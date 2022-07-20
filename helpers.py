from __future__ import division
#torch
import torch
import torch.nn.functional as F
import torch.distributed as torch_dist

import numpy as np
import time
import os
import logging
from pathlib import Path
from importlib import reload
import sys

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
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
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)


def torch_barrier():
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    ALL PROCESSES has the averaged results.
    """
    if torch_dist.is_initialized():
        world_size = torch_dist.get_world_size()
        if world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            torch.distributed.all_reduce(reduced_inp)
            torch.distributed.barrier()
        return reduced_inp / world_size
    return inp

def print_loss_dict(loss, save=None):
    s = ''
    for key in sorted(loss.keys()):
        s += '{}: {:.6f}\n'.format(key, loss[key])
    print (s)
    if save is not None:
        with open(save, 'w') as f:
            f.write(s)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(output_dir, cfg_name, phase='train'):
    root_output_dir = Path(output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    # reset logging
    logging.shutdown()
    reload(logging)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(final_output_dir)

def poly_lr(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    # optimizer.param_groups[0]['lr'] = lr
    for param_group in optimizer.param_groups:
        if 'lr_ratio' in param_group:
            param_group['lr'] = lr * param_group['lr_ratio']
        else:
            param_group['lr'] = lr
    return lr

def const_lr(optimizer, base_lr, max_iters, cur_iters):
    # optimizer.param_groups[0]['lr'] = base_lr
    for param_group in optimizer.param_groups:
        if 'lr_ratio' in param_group:
            param_group['lr'] = base_lr * param_group['lr_ratio']
        else:
            param_group['lr'] = base_lr
    return base_lr

def stair_lr(optimizer, base_lr, max_iters, cur_iters):
    #         0, 180
    ratios = [1, 0.1]
    progress = cur_iters / float(max_iters)
    if progress < 0.9:
        ratio = ratios[0]
    else:
        ratio = ratios[-1]
    lr = base_lr * ratio
    # optimizer.param_groups[0]['lr'] = lr
    for param_group in optimizer.param_groups:
        if 'lr_ratio' in param_group:
            param_group['lr'] = lr * param_group['lr_ratio']
        else:
            param_group['lr'] = lr
    return lr

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

STR_DICT = {
    'poly': poly_lr,
    'const': const_lr,
    'stair': stair_lr
}



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time

code_begin_time = time.time()
memorize_iter_time = list()
memorize_iter_time.append(code_begin_time)

def progress_bar(current, total, current_epoch, start_epoch, end_epoch, mode=None, msg=None):
    # global last_time, begin_time, code_begin_time, runing_weight
    global last_time, begin_time, memorize_iter_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  E: %d' % current_epoch)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if mode:
        memorize_iter_num = 1000
        total_time_from_code_begin = time.time()
        memorize_iter_time.append(total_time_from_code_begin)
        if len(memorize_iter_time) > memorize_iter_num:
            memorize_iter_time.pop(0)
        remain_iters = ((end_epoch-current_epoch)*total) - (current+1)
        eta = (memorize_iter_time[-1] - memorize_iter_time[0]) / (len(memorize_iter_time) - 1) * remain_iters
        L.append(' | ETA: %s' % format_time(eta))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def load_NoPrefix(path, length):
    # load dataparallel wrapped model properly
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[length:] # remove `Scale.`
        new_state_dict[name] = v
    return new_state_dict


def get_model_name(cfg):
    names = {1: 's1_OTVM_alpha',
             2: 's2_OTVM_alpha',
             3: 's3_OTVM',
             4: 's4_OTVM'}
    return names[cfg.TRAIN.STAGE]



def get_model_trimap(cfg, mode='Test', dilate_kernel=None):
    import models.trimap.model as model_trimap
    if mode == 'Train':
        model = model_trimap.FullModel
    elif mode == 'Test':
        model = model_trimap.FullModel_eval

    hdim = 16

    model_loded = model(eps=0,
                        stage=cfg.TRAIN.STAGE,
                        dilate_kernel=dilate_kernel,
                        hdim=hdim,)
    
    return model_loded

def get_model_alpha(cfg, model_trimap, mode='Test', dilate_kernel=None):
    import models.alpha.model as model_alpha
    if cfg.TRAIN.STAGE == 1:
        model_trimap = None

    if mode == 'Train':
        model = model_alpha.FullModel
    elif mode == 'Test':
        model = model_alpha.EvalModel
    
    model_loded = model(dilate_kernel=dilate_kernel,
                        trimap=model_trimap,
                        stage=cfg.TRAIN.STAGE,)

    return model_loded
