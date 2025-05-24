import time
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

from loss import CwRLoss    
from utils import acc_cov, cost_loss
from models import GAT, GraphSAGE, GCN, GATv2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-mps', action='store_true', default=False, help='disables macOS GPU training')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--layers', type=int, default=2, help='Number of layers')

parser.add_argument('--cost', type=float, default=0.7, help='Cost of rejection')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
parser.add_argument('--model', type=str, default='GAT', help='GNN Architecture')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device = torch.device("cpu")
print('Device: ', device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# Load data
name_data = args.dataset
dataset = Planetoid(root= './data/', name = name_data)
dataset.transform = T.NormalizeFeatures()
data = dataset[0].to(device)

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

if args.model == 'GAT':
    model = GAT(dataset.num_features, dataset.num_classes + 1, args.hidden, args.nb_heads, 1, args.dropout)
elif args.model == 'GraphSAGE' or args.model == 'graphsage':
    model = GraphSAGE(dataset.num_features, dataset.num_classes + 1, args.hidden, args.dropout)
elif args.model == 'GCN' or args.model == 'gcn':
    model = GCN(dataset.num_features, dataset.num_classes + 1, args.hidden, args.dropout)
elif args.model == 'GATv2' or args.model == 'gatv2':
    model = GATv2(dataset.num_features, dataset.num_classes + 1, args.hidden, args.nb_heads, 1, args.dropout)
else:
    raise ValueError(f"Model {args.model} not supported")
optimizer = optim.Adam(model.parameters(), 
                      lr=args.lr, 
                      weight_decay=args.weight_decay)
criterion = CwRLoss(args.cost, dataset.num_classes)
model.to(device)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss_train = criterion(output[data.train_mask], data.y[data.train_mask])
    acc_train, cov_train = acc_cov(output[data.train_mask], data.y[data.train_mask])
    cost_loss_train = cost_loss(output[data.train_mask], data.y[data.train_mask], args.cost)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(data.x, data.edge_index)

    loss_val = criterion(output[data.val_mask], data.y[data.val_mask])
    acc_val, cov_val = acc_cov(output[data.val_mask], data.y[data.val_mask])
    cost_loss_val = cost_loss(output[data.val_mask], data.y[data.val_mask], args.cost)
    print('Epoch: {:04d}'.format(epoch+1),
          '0-d-1 Loss Train: {:.3f}'.format(cost_loss_train.data.item()),
          'Acc Train: {:.1f}'.format(acc_train.data.item() * 100),
          'Cov train: {:.1f}'.format(cov_train.data.item() * 100))
    print('0-d-1 Loss Val: {:.3f}'.format(cost_loss_val.data.item()),
          'Acc Val: {:.1f}'.format(acc_val.data.item() * 100),
          'Cov Val: {:.1f}'.format(cov_val.data.item() * 100),
          'Time: {:.3f}s'.format(time.time() - t))

    return loss_val.data.item()

def compute_test():
    model.eval()
    output = model(data.x, data.edge_index)
    loss_test = criterion(output[data.test_mask], data.y[data.test_mask])
    acc_test, cov_test = acc_cov(output[data.test_mask], data.y[data.test_mask])
    cost_loss_test = cost_loss(output[data.test_mask], data.y[data.test_mask], args.cost)
    print("Input parameters:",
          "Cost (d)= {:.2f}".format(args.cost))
    print("Test set results:",
          "0-d-1 Loss= {:.3f}".format(cost_loss_test.data.item()),
          "Accuracy= {:.2f}".format(acc_test.data.item() * 100),
          "Coverage= {:.2f}".format(cov_test.data.item() * 100))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
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
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
