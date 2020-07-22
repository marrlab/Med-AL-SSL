from utils import AverageMeter
import time
import torch
import torch.nn.functional as F


def get_samples(self, epoch, args, model, unlabeled_loader, number):
    batch_time = AverageMeter()
    samples = None

    end = time.time()

    model.eval()

    for i, (data_x, data_y) in enumerate(unlabeled_loader):
        data_x = data_x.cuda(non_blocking=True)

        with torch.no_grad():
            output, feat = model(data_x)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('{0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  .format(self.uncertainty_sampling_method, epoch, i, len(unlabeled_loader), batch_time=batch_time))

    if self.uncertainty_sampling_method == 'entropy_based':
        return samples.argsort(descending=True)[:number]
    else:
        return samples.argsort()[:number]
