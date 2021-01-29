from datetime import datetime

import numpy as np
import os
import shutil
import math
import random

import torch
import torch.nn as nn
import torchvision

from numpy.random import default_rng
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score, \
    pairwise_distances
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data.dataset_utils import WeaklySupervisedDataset
from model.densenet import densenet121
from model.lenet import LeNet
from model.loss_net import LossNet
from model.resnet import resnet18
from model.resnet_autoencoder import ResnetAutoencoder
from model.simclr_arch import SimCLRArch
from model.wideresnet import WideResNet
from augmentations.randaugment import RandAugmentMC

import torch.nn.functional as F
import torchvision.models as models


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar', best_model_filename='model_best.pth.tar'):
    directory = os.path.join(args.checkpoint_path, args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, best_model_filename))


class AverageMeter(object):
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


class LossPerClassMeter(object):
    def __init__(self, classes_num):
        self.classes_num = classes_num
        self.avg = [0 for _ in range(classes_num)]
        self.sum = [0 for _ in range(classes_num)]
        self.count = [0 for _ in range(classes_num)]

    def reset(self):
        self.avg = [0 for _ in range(self.classes_num)]
        self.sum = [0 for _ in range(self.classes_num)]
        self.count = [0 for _ in range(self.classes_num)]

    def update(self, losses, targets):
        for i in range(self.classes_num):
            self.sum[i] += np.sum(losses[targets == i])
            self.count[i] += np.sum(targets == i)
            # noinspection PyTypeChecker
            self.avg[i] = self.sum[i] / (self.count[i] + 1e-6)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, *self.shape)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


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


def create_loaders(args, labeled_dataset, unlabeled_dataset, test_dataset, labeled_indices, unlabeled_indices, kwargs,
                   unlabeled_subset_num):
    labeled_dataset.indices = labeled_indices
    random.shuffle(unlabeled_indices)
    unlabeled_dataset.indices = unlabeled_indices[:unlabeled_subset_num]

    labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    return labeled_loader, unlabeled_loader, val_loader


def create_base_loader(base_dataset, kwargs, batch_size):
    return DataLoader(dataset=base_dataset, batch_size=batch_size, drop_last=True, shuffle=True, **kwargs)


def random_sampling(unlabeled_indices, number):
    rng = default_rng()
    samples_indices = rng.choice(unlabeled_indices.shape[0], size=number, replace=False)

    return samples_indices


def postprocess_indices(labeled_indices, unlabeled_indices, samples_indices):
    unlabeled_mask = torch.ones(size=(len(unlabeled_indices),), dtype=torch.bool)
    unlabeled_mask[samples_indices] = 0
    labeled_indices = np.hstack([labeled_indices, unlabeled_indices[~unlabeled_mask]])
    unlabeled_indices = unlabeled_indices[unlabeled_mask]

    return labeled_indices, unlabeled_indices


class Metrics:
    def __init__(self):
        self.targets = []
        self.outputs = []
        self.outputs_probs = None

    def add_mini_batch(self, mini_targets, mini_outputs):
        self.targets.extend(mini_targets.tolist())
        self.outputs.extend(torch.argmax(mini_outputs, dim=1).tolist())
        self.outputs_probs = mini_outputs \
            if self.outputs_probs is None else torch.cat([self.outputs_probs, mini_outputs], dim=0)

    def get_metrics(self, average='macro'):
        return precision_recall_fscore_support(self.targets, self.outputs, average=average, zero_division=1)

    def get_report(self, target_names):
        return classification_report(self.targets, self.outputs,
                                     zero_division=1, output_dict=True, target_names=target_names)

    def get_confusion_matrix(self):
        return confusion_matrix(self.targets, self.outputs)

    def get_roc_auc_curve(self):
        self.outputs_probs = torch.softmax(self.outputs_probs, dim=1)
        return roc_auc_score(self.targets, self.outputs_probs.cpu().numpy(), multi_class='ovr')


class NTXent(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXent, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples()

    def mask_correlated_samples(self):
        # noinspection PyTypeChecker
        mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )

        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size

        return loss


class TransformsSimCLR:
    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class TransformFix(object):
    def __init__(self, input_size=32, crop_size=32):
        self.weak = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
            torchvision.transforms.Resize(size=input_size),
            torchvision.transforms.ToTensor(),
        ])
        self.strong = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size, padding=int(crop_size * 0.125), padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            torchvision.transforms.Resize(size=input_size),
            torchvision.transforms.ToTensor(),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong


def create_model_optimizer_scheduler(args, dataset_class, optimizer='adam', scheduler='steplr',
                                     load_optimizer_scheduler=False):
    if args.arch == 'wideresnet':
        model = WideResNet(depth=args.layers,
                           num_classes=dataset_class.num_classes,
                           widen_factor=args.widen_factor,
                           dropout_rate=args.drop_rate)
    elif args.arch == 'densenet':
        model = densenet121(num_classes=dataset_class.num_classes)
    elif args.arch == 'lenet':
        model = LeNet(num_channels=3, num_classes=dataset_class.num_classes,
                      droprate=args.drop_rate, input_size=dataset_class.input_size)
    elif args.arch == 'resnet':
        model = resnet18(num_classes=dataset_class.num_classes, input_size=dataset_class.input_size,
                         drop_rate=args.drop_rate)
    else:
        raise NotImplementedError

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    nesterov=args.nesterov, weight_decay=args.weight_decay)

    if scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    else:
        args.iteration = args.fixmatch_k_img // args.batch_size
        args.total_steps = args.fixmatch_epochs * args.iteration
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, args.fixmatch_warmup * args.iteration, args.total_steps)

    if args.resume:
        if load_optimizer_scheduler:
            model, optimizer, scheduler = resume_model(args, model, optimizer, scheduler)
        else:
            model, _, _ = resume_model(args, model)

    return model, optimizer, scheduler


def create_model_optimizer_simclr(args, dataset_class):
    model = SimCLRArch(num_channels=3,
                       num_classes=dataset_class.num_classes,
                       drop_rate=args.drop_rate, normalize=True, arch=args.simclr_arch,
                       input_size=dataset_class.input_size)

    model = model.cuda()

    if args.simclr_resume:
        model, _, _ = resume_model(args, model)
        args.start_epoch = args.simclr_train_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None

    return model, optimizer, scheduler, args


def create_model_optimizer_autoencoder(args, dataset_class):
    model = ResnetAutoencoder(z_dim=args.autoencoder_z_dim, num_classes=dataset_class.num_classes,
                              drop_rate=args.drop_rate, input_size=dataset_class.input_size)

    model = model.cuda()

    if args.autoencoder_resume:
        model, _, _ = resume_model(args, model)
        args.start_epoch = args.epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    return model, optimizer, args


def create_model_optimizer_loss_net():
    model = LossNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer


def get_loss(args, labeled_class_samples, reduction='mean'):
    if args.loss == 'ce':
        if args.weighted:
            classes_weights = np.clip(np.sum(labeled_class_samples) / np.array(labeled_class_samples),
                                      a_min=1, a_max=50)
            # noinspection PyArgumentList
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(classes_weights).cuda(), reduction=reduction)
        else:
            criterion = nn.CrossEntropyLoss(reduction=reduction).cuda()
    else:
        if reduction == 'mean':
            criterion = FocalLoss(gamma=2, alpha=0.25, reduction=True)
        else:
            criterion = FocalLoss(gamma=2, alpha=0.25, reduction=False)

    return criterion


def loss_module_objective_func(pred, target, margin=1.0, reduction='mean'):
    assert len(pred) % 2 == 0, 'the batch size is not even.'
    assert pred.shape == pred.flip(0).shape

    pred = (pred - pred.flip(0))[:len(pred) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    indicator_func = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - indicator_func * pred, min=0))
        loss = loss / pred.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - indicator_func * pred, min=0)
    else:
        loss = None
        NotImplementedError()

    return loss


def resume_model(args, model, optimizer=None, scheduler=None):
    if 'simclr' in args.name:
        name = f"{args.dataset}@{args.arch}@{'simclr'}"
    elif 'auto_encoder' in args.name or 'autoencoder' in args.name:
        name = f"{args.dataset}@{args.arch}@{'auto_encoder'}"
    else:
        name = args.name
    file = os.path.join(args.checkpoint_path, name, 'model_best.pth.tar')
    if os.path.isfile(file):
        print("=> loading checkpoint '{}'".format(file))
        checkpoint = torch.load(file)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint (epoch {0})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{0}'".format(file))

    return model, optimizer, scheduler


def set_model_name(args):
    if args.weak_supervision_strategy == 'semi_supervised':
        name = f"{args.dataset}@{args.arch}@{args.semi_supervised_method}"
    elif args.weak_supervision_strategy == 'active_learning':
        name = f"{args.dataset}@{args.arch}@{args.uncertainty_sampling_method}"
    else:
        name = f"{args.dataset}@{args.arch}@{args.weak_supervision_strategy}"

    name = f'{name}{f"_{args.semi_supervised_uncertainty_method}" if "_with_al" in name else ""}'
    name = f'{name}{"_pretrained" if args.load_pretrained else ""}'
    name = f'{name}{"_k_medoids_100" if args.k_medoids else ""}'
    name = f'{name}{"_novel_class_detection" if args.novel_class_detection else ""}'
    name = f'{name}{f"_{args.semi_supervised_init}" if args.semi_supervised_init is not None else ""}'

    return name


def perform_sampling(args, uncertainty_sampler, epoch, model, train_loader, unlabeled_loader, dataset_class,
                     labeled_indices, unlabeled_indices, labeled_dataset, unlabeled_dataset, test_dataset, kwargs,
                     current_labeled):
    print(args.weak_supervision_strategy)
    if args.weak_supervision_strategy == 'active_learning':
        samples_indices = uncertainty_sampler.get_samples(epoch, args, model,
                                                          train_loader,
                                                          unlabeled_loader,
                                                          number=dataset_class.add_labeled)

        print(f'Uncertainty Sampling\t '
              f'Current labeled ratio: {current_labeled + args.add_labeled}\t'
              f'Model Reset')
    elif args.weak_supervision_strategy == 'random_sampling':
        samples_indices = random_sampling(unlabeled_indices, number=dataset_class.add_labeled)

        print(f'Random Sampling\t '
              f'Current labeled ratio: {current_labeled + args.add_labeled}\t'
              f'Model Reset')

    else:
        samples_indices = uncertainty_sampler.get_samples(epoch, args, model,
                                                          train_loader,
                                                          unlabeled_loader,
                                                          number=dataset_class.add_labeled)

        print(f'Semi Supervised with Active Learning Sampling\t '
              f'Current labeled ratio: {current_labeled + args.add_labeled}\t'
              f'Model Reset')

    labeled_indices, unlabeled_indices = postprocess_indices(labeled_indices, unlabeled_indices,
                                                             samples_indices)

    if args.oversampling:
        labeled_indices = oversampling_indices(labeled_indices,
                                               np.array(labeled_dataset.targets)[labeled_indices])

    train_loader, unlabeled_loader, val_loader = create_loaders(args, labeled_dataset, unlabeled_dataset,
                                                                test_dataset, labeled_indices,
                                                                unlabeled_indices, kwargs,
                                                                dataset_class.unlabeled_subset_num)

    return train_loader, unlabeled_loader, val_loader, labeled_indices, unlabeled_indices


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    # noinspection PyTypeChecker
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


# noinspection PyTypeChecker
def oversampling_indices(indices, targets):
    oversampled_indices = []
    num_classes = []
    masks = []

    for t in np.unique(targets).tolist():
        mask = targets == t
        masks.append(mask)
        num_classes.append(np.sum(mask))

    max_class = np.amax(num_classes)

    for i, t in enumerate(np.unique(targets).tolist()):
        factor = int(max_class / num_classes[i])
        oversampled_indices.extend(np.tile(indices[masks[i]], factor).tolist())

    return np.array(oversampled_indices)


def merge(base_dataset, merge_classes):
    base_targets = np.array(base_dataset.targets)
    base_classes = base_dataset.classes
    base_class_to_idx = base_dataset.class_to_idx

    for m in merge_classes:
        class_idx = []
        for c in m:
            class_idx.append(base_class_to_idx[c])
        class_idx = sorted(class_idx, reverse=True)
        min_i = class_idx[-1]

        class_name = base_classes[min_i]

        for i in class_idx[:-1]:
            base_targets[base_targets == i] = min_i
            class_name += '_'

            for j in range(i + 1, len(base_classes)):
                base_targets[base_targets == j] = j - 1
                base_class_to_idx[base_classes[j]] = j - 1

            class_name += base_classes[i]
            del base_class_to_idx[base_classes[i]]
            del base_classes[i]

        base_class_to_idx[class_name] = base_class_to_idx.pop(base_classes[min_i])
        base_classes[min_i] = class_name

    base_dataset.targets = base_targets
    base_dataset.classes = base_classes
    base_dataset.class_to_idx = base_class_to_idx

    return base_dataset


def remove(base_dataset, classes_to_remove):
    base_targets = np.array(base_dataset.targets)
    base_samples = np.array(base_dataset.samples)
    base_imgs = np.array(base_dataset.imgs)
    classes_to_remove = np.array(classes_to_remove)
    base_classes = base_dataset.classes
    base_class_to_idx = base_dataset.class_to_idx
    base_samples = base_samples[~np.isin(base_targets, classes_to_remove)]
    base_imgs = base_imgs[~np.isin(base_targets, classes_to_remove)]
    base_targets = base_targets[~np.isin(base_targets, classes_to_remove)]

    for r in np.sort(classes_to_remove)[::-1]:
        del base_class_to_idx[base_classes[r]]
        del base_classes[r]

    for i, t in enumerate(np.unique(base_targets)):
        base_targets[base_targets == t] = i
        base_class_to_idx[base_classes[i]] = i

    base_dataset.samples = base_samples.tolist()
    base_dataset.targets = base_targets.tolist()
    base_dataset.imgs = base_imgs.tolist()
    base_dataset.classes = base_classes
    base_dataset.class_to_idx = base_class_to_idx

    return base_dataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, reduction=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = reduction

    def forward(self, outputs, data_y):
        logpt = - F.cross_entropy(outputs, data_y, reduction='none')
        pt = torch.exp(logpt)

        # noinspection PyTypeChecker
        focal_loss = -((1 - pt) ** self.gamma) * logpt

        balanced_focal_loss = self.alpha * focal_loss

        if self.size_average:
            return balanced_focal_loss.mean()
        else:
            return balanced_focal_loss


def load_pretrained(model):
    model_dict = model.state_dict()
    resnet18_pretrained_dict = models.resnet18(pretrained=True).state_dict()

    for key in list(model_dict.keys()):
        if 'linear' in key or 'conv1.weight' == key:
            continue
        model_dict[key] = resnet18_pretrained_dict[key.replace('shortcut', 'downsample')]

    model.load_state_dict(model_dict)

    return model


def class_wise_random_sample(targets, n=1, seed=9999):
    targets = np.array(targets)
    indices = np.arange(len(targets))
    rng = default_rng(seed=seed)

    labeled_indices = []

    for i in np.unique(targets):
        indices_cls = indices[targets == i]
        labeled_indices.extend(rng.choice(indices_cls.shape[0], size=n, replace=False).tolist())

    return labeled_indices, indices[~np.isin(indices, labeled_indices)]


def k_medoids_init(base_dataset, k_medoids_model, transform_test, mean, std, seed, n, k_medoids_n_clusters):
    k_medoids_dataset = WeaklySupervisedDataset(base_dataset, range(len(base_dataset)), transform=transform_test,
                                                mean=mean, std=std)
    k_medoids_loader = DataLoader(dataset=k_medoids_dataset, batch_size=128, shuffle=True)
    k_medoids_model.eval()

    features_h = None

    with torch.no_grad():
        for i, (data_x, data_y) in enumerate(k_medoids_loader):
            data_x = data_x.cuda(non_blocking=True)

            h = k_medoids_model.forward_encoder(data_x)
            features_h = h if features_h is None else torch.cat([features_h, h], dim=0)
            print('K-medoids features: [{0}/{1}]'.format(i, len(k_medoids_loader)))

    features_h = features_h.cpu().numpy()
    dist_mat = pairwise_distances(features_h)

    from sklearn_extra.cluster import KMedoids
    k_medoids_clusterer = KMedoids(n_clusters=k_medoids_n_clusters, metric='precomputed', random_state=seed)
    k_medoids = k_medoids_clusterer.fit(dist_mat)

    indices = np.arange(len(base_dataset))
    labeled_indices = []
    samples_per_cluster = int(n / k_medoids_n_clusters)

    for index in k_medoids.medoid_indices_:
        labeled_indices.extend(np.argsort(dist_mat[index])[:samples_per_cluster])

    labeled_indices = np.unique(k_medoids.medoid_indices_)

    return labeled_indices, indices[~np.isin(indices, labeled_indices)]


def print_args(args):
    print('Arguments:\n'
          f'Model name: {args.name}\t'
          f'Epochs: {args.epochs}\t'
          f'Batch Size: {args.batch_size}\n'
          f'Architecture: {args.arch}\t'
          f'Weak Supervision Strategy: {args.weak_supervision_strategy}\n'
          f'Uncertainty Sampling Method: {args.uncertainty_sampling_method}\t'
          f'Semi Supervised Method: {args.semi_supervised_method}\n'
          f'Dataset root: {args.root}')


def store_logs(args, logs_df, log_type='al_cycles'):
    if log_type == 'epoch_wise':
        filename = '{0}-{1}-seed:{2}-epoch'.format(datetime.now().strftime("%d.%m.%Y"), args.name, args.seed)
    elif log_type == 'ae_loss':
        filename = '{0}-{1}-ae-loss'.format(datetime.now().strftime("%d.%m.%Y"), args.name)
    elif log_type == 'novel_class':
        filename = '{0}-{1}-seed:{2}-class-nums'.format(datetime.now().strftime("%d.%m.%Y"), args.name, args.seed)
    else:
        filename = '{0}-{1}-seed:{2}'.format(datetime.now().strftime("%d.%m.%Y"), args.name, args.seed)

    logs_df.to_csv(os.path.join(args.log_path, filename))
