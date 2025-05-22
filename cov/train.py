from __future__ import division
from __future__ import print_function

import os
import csv
import glob
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.manifold import TSNE

from utils import load_data, accuracy, sel_loss, sel_accuracy, true_cov, find_tres
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import models
import torch_geometric.transforms as T
from models import GAT, GCN, GraphSAGE, GATv2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--layers', type=int, default=2, help='Number of layers')

parser.add_argument('--coverage', type=float, default=0.8, help='Coverage')
parser.add_argument('--lamda', type=int, default=32, help='Lambda')
parser.add_argument('--alphaloss', type=float, default=0.5, help='Alpha')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--model', type=str, default='GAT', help='GNN Architecture')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ', device)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

name_data = args.dataset
dataset = Planetoid(root= '../../data/' , name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

if args.model == 'GAT':
    model = GAT(args.layers, dataset.num_features, dataset.num_classes, args.hidden, args.nb_heads, args.nb_heads, args.dropout)
elif args.model == 'GCN':
    model = GCN(dataset.num_features, dataset.num_classes, args.hidden, args.dropout)
elif args.model == 'GraphSAGE':
    model = GraphSAGE(dataset.num_features, dataset.num_classes, args.hidden, args.dropout)
elif args.model == 'GATv2':
    model = GATv2(dataset.num_features, dataset.num_classes, args.hidden, args.nb_heads, args.nb_heads, args.dropout)
else:
    raise ValueError('Invalid model name')

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
model.to(device)
data = dataset[0].to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss_train = sel_loss(output[0][data.train_mask], output[1][data.train_mask], data.y[data.train_mask], args.coverage, args.lamda, args.alphaloss)
    sel_acc_train = sel_accuracy(output[0][data.train_mask], data.y[data.train_mask])
    true_cov_train = true_cov(output[0][data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(data.x, data.edge_index)

    loss_val = sel_loss(output[0][data.val_mask], output[1][data.val_mask], data.y[data.val_mask], args.coverage, args.lamda, args.alphaloss)
    sel_acc_val = sel_accuracy(output[0][data.val_mask], data.y[data.val_mask])
    true_cov_val = true_cov(output[0][data.val_mask], data.y[data.val_mask])
    print('Epoch: {:04d}'.format(epoch+1),
          'Loss Train: {:.4f}'.format(loss_train.data.item()),
          'Sel Acc Train: {:.4f}'.format(sel_acc_train.data.item()),
          'Cov Train: {:.4f}'.format(true_cov_train.data.item()),
          'Time: {:.4f}s'.format(time.time() - t))
    
    print('Epoch: {:04d}'.format(epoch+1),
          'Loss Val: {:.4f}'.format(loss_val.data.item()),
          'Sel Acc Val: {:.4f}'.format(sel_acc_val.data.item()),
          'Cov Val: {:.4f}'.format(true_cov_val.data.item()),
          'Time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()

def compute_test():
    model.eval()
    output = model(data.x, data.edge_index)
    loss_test = sel_loss(output[0][data.test_mask], output[1][data.test_mask], data.y[data.test_mask], args.coverage, args.lamda, args.alphaloss)
    acc_test = accuracy(output[0][data.test_mask], data.y[data.test_mask])
    tres = find_tres(output[0][data.test_mask], args.coverage)
    sel_acc_test = sel_accuracy(output[0][data.test_mask], data.y[data.test_mask], t=tres)
    true_cov_test = true_cov(output[0][data.test_mask], data.y[data.test_mask], t=tres)
    
    print("Test set results:")
    print("Loss = {:.4f}".format(loss_test.data.item()),
        "Accuracy = {:.4f}".format(acc_test.data.item()))
    print("Treshold = ", round(tres, 3),
        "Selective Accuracy = {:.4f}".format(sel_acc_test.data.item()),
        "Coverage = {:.4f}".format(true_cov_test.data.item()))
    return true_cov_test.data.item()*100, round(sel_acc_test.data.item()*100, 2)

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    # torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
cov, acc = compute_test()

with open(f'{args.dataset}.csv', mode='a') as file:
    writer = csv.writer(file)
    writer.writerow([args.dataset, args.model, args.layers, args.coverage, cov, acc])