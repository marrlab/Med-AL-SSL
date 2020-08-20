from utils import AverageMeter
import time
import torch
import numpy as np

"""
Bayesian Active Learning by Disagreement (BALD) extension

Implementation of:
Deep Bayesian Active Learning with Image Data:
https://arxiv.org/abs/1703.02910
"""


class UncertaintySamplingMCDropout:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def entropy(probs):
        log_probs = torch.log(probs)
        entropy = torch.sum(-probs * log_probs, dim=1)

        return entropy

    def get_samples(self, epoch, args, model, _, unlabeled_loader, number):
        batch_time = AverageMeter()
        all_score = None
        all_entropy = None
        targets = None

        end = time.time()

        model.train()

        for j in range(args.mc_dropout_iterations):
            scores = None

            for i, (data_x, data_y) in enumerate(unlabeled_loader):
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                with torch.no_grad():
                    output, _, _ = model(data_x)

                scores = output if scores is None else torch.cat([scores, output])
                targets = data_y.cpu().numpy() if targets is None \
                    else np.concatenate([targets, data_y.cpu().numpy().tolist()])

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('{0}\t'
                          'Epoch: [{1}][{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          .format(args.uncertainty_sampling_method, epoch, i, len(unlabeled_loader),
                                  batch_time=batch_time))
            print('\n MC dropout sample: ', j+1)

            all_score = scores if all_score is None else all_score + scores
            all_entropy = self.entropy(scores) if all_entropy is None else all_entropy + self.entropy(scores)

        avg_score = all_score / args.mc_dropout_iterations
        entropy_avg_score = self.entropy(avg_score)

        average_entropy = all_entropy / args.mc_dropout_iterations

        scores = entropy_avg_score - average_entropy

        return scores.argsort(descending=True)[:number]
