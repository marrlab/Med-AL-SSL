import torch
from utils import AverageMeter
import time
import torch.nn.functional as F


class PseudoLabeling:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def get_samples(epoch, args, model, unlabeled_loader, number):
        batch_time = AverageMeter()
        samples = None
        samples_targets = None

        end = time.time()

        model.eval()

        for i, (data_x, _) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)

            with torch.no_grad():
                output, _, _ = model(data_x)
            score = F.softmax(output, dim=1)
            score = torch.max(score, dim=1)

            samples = score[0] if samples is None else torch.cat([samples, score[0]])
            samples_targets = score[1] if samples_targets is None else torch.cat([samples_targets, score[1]])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('{0}\t'
                      'Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format('Pseudo Labeling', epoch, i, len(unlabeled_loader), batch_time=batch_time))

        samples_targets = samples_targets[samples > args.pseudo_labeling_threshold]
        samples = samples[samples > args.pseudo_labeling_threshold]

        samples_indices = samples.argsort(descending=True)[:number]
        samples_targets = samples_targets[samples_indices]

        return samples_indices, samples_targets
