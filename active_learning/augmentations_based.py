from utils import AverageMeter
import time, sys
import torch
import numpy as np

"""
Mode based uncertainty sampling classification

Implementation of:
Multiclass Deep Active Learning for Detecting Red Blood Cell Subtypes in Brightfield Microscopy:
https://link.springer.com/chapter/10.1007/978-3-030-32239-7_76
"""


class UncertaintySamplingAugmentationBased:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def get_samples(epoch, args, model, _, unlabeled_loader, number):
        batch_time = AverageMeter()
        end = time.time()
        model.eval()

        all_max_classes = None
        targets = None

        for j in range(args.augmentations_based_iterations):

            max_classes = None
            targets = None

            for i, (data_x, data_y) in enumerate(unlabeled_loader):
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                with torch.no_grad():
                    if args.weak_supervision_strategy == 'semi_supervised_active_learning':
                        output = model.forward_encoder_classifier(data_x)
                    else:
                        output, _, _ = model(data_x)

                output = torch.argmax(output, dim=1)

                max_classes = output if max_classes is None else torch.cat([max_classes, output])
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
            print('\n Augmentations based sample: ', j+1)

            all_max_classes = torch.unsqueeze(max_classes, dim=1) if all_max_classes is None else \
                torch.cat([all_max_classes, torch.unsqueeze(max_classes, dim=1)], dim=1)

        all_modes = torch.mode(all_max_classes, dim=1).values
        scores = torch.zeros(all_max_classes.size(0))

        for j in range(all_max_classes.size(0)):
            scores[j] = torch.sum(all_max_classes[j] == all_modes[j])

        original_stdout = sys.stdout

        with open(f'output_{args.semi_supervised_method}.txt', 'w') as f:
            sys.stdout = f
            print(targets.tolist())
            print(scores[scores.argsort()[:number]].cpu().numpy().tolist())
            print(np.array(targets)[scores.argsort()[:number].cpu().numpy()].tolist())
            sys.stdout = original_stdout

        return scores.argsort()[:number]
