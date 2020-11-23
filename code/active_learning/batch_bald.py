from utils import AverageMeter
import time
import torch
import numpy as np
from toma import toma

"""
Implementation of:
BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning
https://arxiv.org/abs/1906.08158

Courtesy to:
https://github.com/BlackHC/batchbald_redux
"""


class JointEntropy:
    def compute(self) -> torch.Tensor:
        raise NotImplementedError()

    def add_variables(self, probs_N_K_C: torch.Tensor) -> 'JointEntropy':
        raise NotImplementedError()

    def compute_batch(self,
                      probs_B_K_C: torch.Tensor,
                      output_entropies_B=None) -> torch.Tensor:
        raise NotImplementedError()


class ExactJointEntropy(JointEntropy):
    joint_probs_M_K: torch.Tensor

    def __init__(self, joint_probs_M_K: torch.Tensor):
        self.joint_probs_M_K = joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> 'ExactJointEntropy':
        return ExactJointEntropy(torch.ones((1, K), device=device,
                                            dtype=dtype))

    def compute(self) -> torch.Tensor:
        probs_M = torch.mean(self.joint_probs_M_K, dim=1, keepdim=False)
        nats_M = -torch.log(probs_M) * probs_M
        entropy = torch.sum(nats_M)
        return entropy

    def add_variables(self, probs_N_K_C: torch.Tensor) -> 'ExactJointEntropy':
        assert self.joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

        N, K, C = probs_N_K_C.shape
        joint_probs_K_M_1 = self.joint_probs_M_K.t()[:, :, None]

        # Using lots of memory.
        for i in range(N):
            probs_i__K_1_C = probs_N_K_C[i][:, None, :].to(joint_probs_K_M_1,
                                                           non_blocking=True)
            joint_probs_K_M_C = joint_probs_K_M_1 * probs_i__K_1_C
            joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

        self.joint_probs_M_K = joint_probs_K_M_1.squeeze(2).t()
        return self

    def compute_batch(self,
                      probs_B_K_C: torch.Tensor,
                      output_entropies_B=None):
        assert self.joint_probs_M_K.shape[1] == probs_B_K_C.shape[1]

        B, K, C = probs_B_K_C.shape
        M = self.joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B,
                                             dtype=probs_B_K_C.dtype,
                                             device=probs_B_K_C.device)

        @toma.execute.chunked(probs_B_K_C,
                              initial_step=1024,
                              dimension=0)
        def chunked_joint_entropy(chunked_probs_b_K_C: torch.Tensor,
                                  start: int, end: int):
            b = chunked_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty((b, M, C),
                                      dtype=self.joint_probs_M_K.dtype,
                                      device=self.joint_probs_M_K.device)
            for i in range(b):
                torch.matmul(self.joint_probs_M_K,
                             chunked_probs_b_K_C[i].to(self.joint_probs_M_K, non_blocking=True), out=probs_b_M_C[i])
            probs_b_M_C /= K

            output_entropies_B[start:end].copy_(torch.sum(
                -torch.log(probs_b_M_C) * probs_b_M_C, dim=(1, 2)),
                                                non_blocking=True)

        return output_entropies_B


def batch_multi_choices(probs_b_C, M: int):
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))

    choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)

    choices_b_M = choices.reshape(list(probs_b_C.shape[:-1]) + [M])
    return choices_b_M


def gather_expand(data, dim, index):
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


class SampledJointEntropy(JointEntropy):
    sampled_joint_probs_M_K: torch.Tensor

    def __init__(self, sampled_joint_probs_M_K: torch.Tensor):
        self.sampled_joint_probs_M_K = sampled_joint_probs_M_K

    @staticmethod
    def empty(K: int, device=None, dtype=None) -> 'SampledJointEntropy':
        return SampledJointEntropy(
            torch.ones((1, K), device=device, dtype=dtype))

    @staticmethod
    def sample(probs_N_K_C: torch.Tensor, M: int) -> 'SampledJointEntropy':
        K = probs_N_K_C.shape[1]

        S = M // K

        choices_N_K_S = batch_multi_choices(probs_N_K_C, S).long()

        expanded_choices_N_1_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_1_C = probs_N_K_C[:, :, None, :]

        probs_N_K_K_S = gather_expand(expanded_probs_N_K_1_C,
                                      dim=-1,
                                      index=expanded_choices_N_1_K_S)
        probs_K_K_S = torch.exp(
            torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))

        samples_M_K = samples_K_M.t()
        return SampledJointEntropy(samples_M_K)

    def compute(self) -> torch.Tensor:
        sampled_joint_probs_M = torch.mean(self.sampled_joint_probs_M_K,
                                           dim=1,
                                           keepdim=False)
        nats_M = -torch.log(sampled_joint_probs_M)
        entropy = torch.mean(nats_M)
        return entropy

    # noinspection PyMethodOverriding
    def add_variables(self, probs_N_K_C: torch.Tensor,
                      M2: int) -> 'SampledJointEntropy':

        K = probs_N_K_C.shape[1]
        assert self.sampled_joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

        sample_K_M1_1 = self.sampled_joint_probs_M_K.t()[:, :, None]

        new_sample_M2_K = self.sample(probs_N_K_C, M2).sampled_joint_probs_M_K
        new_sample_K_1_M2 = new_sample_M2_K.t()[:, None, :]

        merged_sample_K_M1_M2 = sample_K_M1_1 * new_sample_K_1_M2
        merged_sample_K_M = merged_sample_K_M1_M2.reshape((K, -1))

        self.sampled_joint_probs_M_K = merged_sample_K_M.t()

        return self

    def compute_batch(self,
                      probs_B_K_C: torch.Tensor,
                      output_entropies_B=None):
        assert self.sampled_joint_probs_M_K.shape[1] == probs_B_K_C.shape[1]

        B, K, C = probs_B_K_C.shape
        M = self.sampled_joint_probs_M_K.shape[0]

        if output_entropies_B is None:
            output_entropies_B = torch.empty(B,
                                             dtype=probs_B_K_C.dtype,
                                             device=probs_B_K_C.device)

        @toma.execute.chunked(probs_B_K_C,
                              initial_step=1024,
                              dimension=0)
        def chunked_joint_entropy(chunked_probs_b_K_C: torch.Tensor,
                                  start: int, end: int):
            b = chunked_probs_b_K_C.shape[0]

            probs_b_M_C = torch.empty(
                (b, M, C),
                dtype=self.sampled_joint_probs_M_K.dtype,
                device=self.sampled_joint_probs_M_K.device)
            for i in range(b):
                torch.matmul(self.sampled_joint_probs_M_K,
                             probs_B_K_C[i].to(self.sampled_joint_probs_M_K,
                                               non_blocking=True),
                             out=probs_b_M_C[i])
            probs_b_M_C /= K

            q_1_M_1 = self.sampled_joint_probs_M_K.mean(dim=1,
                                                        keepdim=True)[None]

            output_entropies_B[start:end].copy_(
                torch.sum(-torch.log(probs_b_M_C) * probs_b_M_C / q_1_M_1,
                          dim=(1, 2)) / M,
                non_blocking=True)

        return output_entropies_B


class DynamicJointEntropy(JointEntropy):
    inner: JointEntropy
    probs_max_N_K_C: torch.Tensor
    N: int
    M: int

    def __init__(self, M: int, max_N: int, K: int, C: int, dtype=None, device=None):
        self.M = M
        self.N = 0
        self.max_N = max_N

        self.inner = ExactJointEntropy.empty(K, dtype=dtype, device=device)
        self.probs_max_N_K_C = torch.empty((max_N, K, C), dtype=dtype, device=device)

    def add_variables(self, probs_N_K_C: torch.Tensor) -> 'DynamicJointEntropy':
        C = self.probs_max_N_K_C.shape[2]
        add_N = probs_N_K_C.shape[0]

        assert self.probs_max_N_K_C.shape[0] >= self.N + add_N
        assert self.probs_max_N_K_C.shape[2] == C

        self.probs_max_N_K_C[self.N:self.N+add_N] = probs_N_K_C
        self.N += add_N

        num_exact_samples = C**self.N
        if num_exact_samples > self.M:
            self.inner = SampledJointEntropy.sample(self.probs_max_N_K_C[:self.N], self.M)
        else:
            self.inner.add_variables(probs_N_K_C)

        return self

    def compute(self) -> torch.Tensor:
        return self.inner.compute()

    def compute_batch(self,
                      probs_B_K_C: torch.Tensor,
                      output_entropies_B=None) -> torch.Tensor:
        return self.inner.compute_batch(probs_B_K_C, output_entropies_B)


class UncertaintySamplingBatchBald:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def compute_conditional_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
        N, K, C = probs_N_K_C.shape

        entropies_N = torch.empty(N, dtype=torch.double)

        @toma.execute.chunked(probs_N_K_C, 1024)
        def compute(probs_n_K_C, start: int, end: int):
            nats_n_K_C = probs_n_K_C * torch.log(probs_n_K_C)
            nats_n_K_C[probs_n_K_C == 0] = 0.

            entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)

        return entropies_N

    @staticmethod
    def compute_entropy(probs_N_K_C: torch.Tensor) -> torch.Tensor:
        N, K, C = probs_N_K_C.shape

        entropies_N = torch.empty(N, dtype=torch.double)

        @toma.execute.chunked(probs_N_K_C, 1024)
        def compute(probs_n_K_C, start: int, end: int):
            mean_probs_n_C = probs_n_K_C.mean(dim=1)
            nats_n_C = mean_probs_n_C * torch.log(mean_probs_n_C)
            nats_n_C[mean_probs_n_C == 0] = 0.

            entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))

        return entropies_N

    def get_batchbald_batch(self, probs_N_K_C: torch.Tensor,
                            batch_size: int,
                            num_samples: int,
                            dtype=None,
                            device=None):
        N, K, C = probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []
        candidate_scores = []

        conditional_entropies_N = self.compute_conditional_entropy(probs_N_K_C)

        batch_joint_entropy = DynamicJointEntropy(num_samples, batch_size - 1, K, C, dtype=dtype, device=device)

        scores_N = torch.empty(N, dtype=torch.double, device=device)

        for i in range(batch_size):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(
                    probs_N_K_C[latest_index:latest_index + 1])

            shared_conditional_entropies = conditional_entropies_N[
                candidate_indices].sum()

            batch_joint_entropy.compute_batch(probs_N_K_C,
                                              output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditional_entropies
            scores_N[candidate_indices] = -float('inf')

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return candidate_scores, candidate_indices

    def get_samples(self, epoch, args, model, _, unlabeled_loader, number):
        batch_time = AverageMeter()
        targets = None
        all_scores = None

        end = time.time()

        model.train()

        for j in range(args.mc_dropout_iterations):
            scores = None
            for i, (data_x, data_y) in enumerate(unlabeled_loader):
                data_x = data_x.cuda(non_blocking=True)
                data_y = data_y.cuda(non_blocking=True)

                with torch.no_grad():
                    if args.weak_supervision_strategy == 'semi_supervised_active_learning':
                        output = torch.softmax(model.forward_encoder_classifier(data_x), dim=1)
                    else:
                        output = torch.softmax(model(data_x), dim=1)

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
            if all_scores is None:
                all_scores = scores
            elif all_scores.dim() > 2:
                all_scores = torch.cat([all_scores, scores.unsqueeze(dim=1)], dim=1)
            else:
                all_scores = torch.cat([all_scores.unsqueeze(dim=1), scores.unsqueeze(dim=1)], dim=1)

            print('\n BatchBald sample: ', j+1)

        scores, indices = self.get_batchbald_batch(all_scores, batch_size=number,
                                                   num_samples=args.mc_dropout_iterations)

        return indices
