import os
import torch
import shutil
from torch.utils.data import DataLoader, SubsetRandomSampler


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % args.name + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_loaders(args, base_dataset, test_dataset, labeled_idx, unlabeled_idx, kwargs):
    labeled_loader = DataLoader(dataset=base_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                sampler=SubsetRandomSampler(labeled_idx),
                                **kwargs)

    unlabeled_loader = DataLoader(dataset=base_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  sampler=SubsetRandomSampler(unlabeled_idx),
                                  **kwargs)

    val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             **kwargs)

    return labeled_loader, unlabeled_loader, val_loader
