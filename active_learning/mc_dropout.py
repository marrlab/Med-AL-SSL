from utils import AverageMeter
import time
import torch


def get_samples(epoch, args, model, unlabeled_loader, number):
    batch_time = AverageMeter()
    all_samples = []

    end = time.time()

    model.train()

    for j in range(args.mc_dropout_iterations):
        sample = None

        for i, (data_x, data_y) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)

            with torch.no_grad():
                output, feat = model(data_x)

            sample = output if sample is None else torch.cat([sample, output])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('{0}\t'
                      'Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format(args.uncertainty_sampling_method, epoch, i, len(unlabeled_loader), batch_time=batch_time))
        print('\n MC dropout sample: ', j)
        all_samples = output if all_samples is None else torch.cat([all_samples, sample], dim=1)

    all_samples = torch.std(all_samples, dim=1)

    return all_samples.argsort(descending=True)[:number]
