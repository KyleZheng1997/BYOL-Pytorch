import torch
from util.torch_dist_sum import *
from data.imagenet import *
from data.augmentation import *
from util.meter import *
import torch.nn as nn
import torch.nn.functional as F
from network.byol import Byol
import time
import math
from util.LARS import LARS
# from util.LARC import LARC


epochs = 300
warm_up = 10

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = 0.5 * base_lr * (1 + math.cos(1.0 * T / total_iters * math.pi))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(train_loader, model, local_rank, rank, optimizer, epoch, lr):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    iteration_per_epoch = train_loader.__len__()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, lr, i, iteration_per_epoch)
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)
    
        cur_iter = iteration_per_epoch * epoch + i
        max_iter = epochs * iteration_per_epoch

        # compute output
        q, t = model(img1, img2, cur_iter, max_iter)
        loss = 2 - 2 * (q * t.detach()).sum(-1)
        loss = loss.mean()
        

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), img1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # model.average_gradients()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0 and rank == 0:
            progress.display(i)


def main():
    
    from util.dist_init import dist_init
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    rank, local_rank, world_size = dist_init()
    batch_size = 128 # single gpu
    num_workers = 8

    _LR_PRESETS = {40: 0.45, 100: 0.45, 300: 0.3, 1000: 0.2}
    _WD_PRESETS = {40: 1e-6, 100: 1e-6, 300: 1e-6, 1000: 1.5e-6}
    _EMA_PRESETS = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}

    lr = _LR_PRESETS[epochs] * (batch_size * world_size / 256)
    wd = _WD_PRESETS[epochs]
    
    model = Byol(base_momentum=_EMA_PRESETS[epochs])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n) and ('encoder_q' in n or 'predictor' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n) and ('encoder_q' in n or 'predictor' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}], 
                                lr=lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)

    model = DDP(model, device_ids=[local_rank])
    torch.backends.cudnn.benchmark = True

    train_dataset = ImagenetContrastive(aug=[byol_aug1, byol_aug2])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, local_rank, rank, optimizer, epoch, lr)
        if rank == 0 and (epoch+1) % 50 == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoints/byol-{}.pth'.format(epoch + 1))

if __name__ == "__main__":
    main()
