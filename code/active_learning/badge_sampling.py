import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from scipy import stats


class UncertaintySamplingBadge:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def get_grad_embedding(model, unlabeled_loader, num_classes, num_unlabeled):
        eDim = model.get_embedding_dim()
        model.eval()
        embedding = np.zeros([num_unlabeled, eDim * num_classes])

        with torch.no_grad():
            for i, (data_x, _) in enumerate(unlabeled_loader):
                data_x = data_x.cuda(non_blocking=True)
                batch_size = unlabeled_loader.batch_size
                idxs = np.arange(i * batch_size, (i * batch_size) + batch_size)

                out, feat = model.forward_embeddings(data_x)
                feat = feat.data.cpu().numpy()
                batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)

                for j in range(data_x.shape[0]):
                    for c in range(num_classes):
                        if c == maxInds[j]:
                            embedding[idxs[j]][eDim * c: eDim * (c + 1)] = deepcopy(feat[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][eDim * c: eDim * (c + 1)] = deepcopy(feat[j]) * (-1 * batchProbs[j][c])

            return embedding

    @staticmethod
    def init_centers(X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('Badge Sampling')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]

            print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        gram = np.matmul(X[indsAll], X[indsAll].T)
        val, _ = np.linalg.eig(gram)
        val = np.abs(val)
        vgt = val[val > 1e-2]
        return indsAll

    def get_samples(self, epoch, args, model, train_loader, unlabeled_loader, num_classes, num_unlabeled, number):
        gradEmbedding = self.get_grad_embedding(model, unlabeled_loader, num_classes, num_unlabeled)
        chosen = self.init_centers(gradEmbedding, number)

        return chosen
