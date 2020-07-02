import torch
import math
import torch.nn.functional as F
from utils import AverageMeter
import time


class UncertaintySampling:
    """
    Active Learning methods to sample for uncertainty
    Credits to: https://github.com/rmunro/pytorch_active_learning
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    @staticmethod
    def least_confidence(probs):
        """
        Returns the uncertainty score of an array using
        least confidence sampling in a 0-1 range where 1 is the most uncertain

        Assumes probability distribution is a pytorch tensor, like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])

        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """

        simple_least_conf = torch.max(probs, dim=1)  # most confident prediction

        return simple_least_conf[0]

    @staticmethod
    def margin_confidence(prob_dist, is_sorted=False):
        """
        Returns the uncertainty score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain

        Assumes probability distribution is a pytorch tensor, like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])

        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not is_sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True)  # sort probs so largest is first

        difference = (prob_dist.data[0] - prob_dist.data[1])  # difference between top two props
        margin_conf = 1 - difference

        return margin_conf.item()

    @staticmethod
    def ratio_confidence(prob_dist, is_sorted=False):
        """
        Returns the uncertainty score of a probability distribution using
        ratio of confidence sampling in a 0-1 range where 1 is the most uncertain

        Assumes probability distribution is a pytorch tensor, like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])

        Keyword arguments:
            prob_dist --  pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not is_sorted:
            prob_dist, _ = torch.sort(prob_dist, descending=True)  # sort probs so largest is first

        ratio_conf = prob_dist.data[1] / prob_dist.data[0]  # ratio between top two props

        return ratio_conf.item()

    @staticmethod
    def entropy_based(prob_dist):
        """
        Returns the uncertainty score of a probability distribution using
        entropy

        Assumes probability distribution is a pytorch tensor, like:
            tensor([0.0321, 0.6439, 0.0871, 0.2369])

        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        log_probs = prob_dist * torch.log2(prob_dist)  # multiply each probability by its base 2 log
        raw_entropy = 0 - torch.sum(log_probs)

        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())

        return normalized_entropy

    def softmax(self, scores, base=math.e):
        """Returns softmax array for array of scores

        Converts a set of raw scores from a model (logits) into a
        probability distribution via softmax.

        The probability distribution will be a set of real numbers
        such that each is in the range 0-1.0 and the sum is 1.0.

        Assumes input is a pytorch tensor: tensor([1.0, 4.0, 2.0, 3.0])

        Keyword arguments:
            prediction -- a pytorch tensor of any positive/negative real numbers.
            base -- the base for the exponential (default e)
        """
        exps = (base ** scores.to(dtype=torch.float))  # exponential for each value in array
        sum_exps = torch.sum(exps)  # sum of all exponentials

        prob_dist = exps / sum_exps  # normalize exponentials
        return prob_dist

    @staticmethod
    def get_samples(epoch, args, model, unlabeled_loader, method, number):
        batch_time = AverageMeter()
        samples = None

        end = time.time()

        model.eval()

        for i, (data_x, _) in enumerate(unlabeled_loader):
            data_x = data_x.cuda(non_blocking=True)

            with torch.no_grad():
                output = model(data_x)
            score = method(F.softmax(output, dim=1))

            samples = score if samples is None else torch.cat([samples, score])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Uncertainty Sampling\t'
                      'Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      .format(epoch, i, len(unlabeled_loader), batch_time=batch_time))

        return samples.argsort()[:number]
