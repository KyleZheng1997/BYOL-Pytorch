import torch
from data.imagenet import *
from data.augmentation import *
from network.head import *
from network.byol import Byol
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from util.meter import *
import time
from util.torch_dist_sum import *
from util.dist_init import dist_init
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()
print(args)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def main():

    rank, local_rank, world_size = dist_init()


    epochs = 80
    batch_size = 128
    num_workers = 8
    lr = 0.1 * batch_size * world_size / 256


    pre_train = Byol()
    pre_train = nn.SyncBatchNorm.convert_sync_batchnorm(pre_train)
    state_dict = torch.load('checkpoints/' + args.checkpoint, map_location='cpu')['model']

    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

    pre_train.load_state_dict(state_dict)

    model = LinearHead(pre_train.encoder_q.net)
    model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = torch.optim.SGD(model.module.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)


    torch.backends.cudnn.benchmark = True

    train_dataset = Imagenet()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)


    test_dataset = Imagenet(mode='val', aug=eval_aug)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=(test_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=test_sampler)


    best_acc = 0
    best_acc5 = 0

    for epoch in range(epochs):
        # ---------------------- Train --------------------------

        train_sampler.set_epoch(epoch)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            train_loader.__len__(),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch)
        )
        end = time.time()

        model.eval()
        for i, (image, label) in enumerate(train_loader):
            data_time.update(time.time() - end)

            image = image.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            
            out = model(image)
            loss = F.cross_entropy(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item())

            if i % 20 == 0 and rank == 0:
                progress.display(i)

        # scheduler.step()

        # ---------------------- Test --------------------------
        model.eval()
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        with torch.no_grad():
            end = time.time()
            for i, (image, label) in enumerate(test_loader):
                
                image = image.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)

                # compute output
                output = model(image)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, label, topk=(1, 5))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        
        sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)

        top1_acc = sum(sum1.float()) / sum(cnt1.float())
        top5_acc = sum(sum5.float()) / sum(cnt5.float())

        best_acc = max(top1_acc, best_acc)
        best_acc5 = max(top5_acc, best_acc5)

        if rank == 0:
            print('Epoch:{} * Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} Best_Acc@5 {best_acc5:.3f}'.format(epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc, best_acc5=best_acc5))

if __name__ == "__main__":
    main()
