import random
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn

def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    output = output[:, :-1]
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# def emp_cov(output, labels):
#     sel = output[:, -1]
#     return torch.mean(sel)


def true_cov(output, labels, t=0.5):
    sel = output[:, -1]
    sel = torch.where(sel>t, 1, 0).double()
    return torch.mean(sel)


def sel_accuracy(output, labels, t=0.5):
    pred = output[:, :-1]
    sel = output[:, -1]
    sel = torch.where(sel>t, 1, 0).double()
    pred = torch.max(pred, 1)[1]
    correct = torch.eq(pred, labels).double()
    correct = torch.count_nonzero(torch.logical_and(correct, sel))
    selected = torch.count_nonzero(sel)
    if selected == 0:
        return torch.tensor(0)
    sel_acc = correct/selected
    return sel_acc

def sel_loss(pred_sel, aux, labels, coverage, lamda, alpha):
    pred, sel = pred_sel[:, :-1], pred_sel[:, -1]
    loss_fn = nn.NLLLoss(reduction='none')
    emp_cov = torch.mean(sel)
    loss1 = torch.mean(loss_fn(pred, labels) * sel) / emp_cov
    coverage = torch.tensor(coverage).double()
    psi = torch.square(torch.maximum((coverage - emp_cov), torch.tensor(0)))
    loss1 += lamda * psi
    loss2 = torch.mean(loss_fn(aux, labels))
    return ((alpha * loss1) + ((1-alpha) * loss2))

def find_tres(output, coverage):
    sel = output[:, -1]
    sel = sel.tolist()
    sel.sort(reverse=True)
    c = int(coverage * len(sel))
    return sel[c]

def flip(labels, x):
  total = len(labels)
  classes = labels.max() + 1
  change = int(total * x)
  idx = [i for i in range(total)]
  random.shuffle(idx)
  idx = idx[:change]
  for i in range(len(idx)):
    idx_i = idx[i]
    labels[idx_i] = random.choice([num for num in range(classes) if num != labels[idx_i]])
  return labels