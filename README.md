# Node Classification with Rejection (NCwR)

This is the implementation of a paper accepted at [**TMLR 2025**](https://openreview.net/pdf?id=4xXJDO8Bvu).

## Project Structure

This repository contains implementations of node classification and two different approaches for integrating reject option:

- **`base/`** - Base implementation with standard GNN training
- **`cost/`** - Cost-based rejection method 
- **`cov/`** - Coverage-based selective prediction method

Each folder contains its own `train.py` with specific implementations and hyperparameters.

## Supported Models

All methods support the following GNN architectures:
- GAT (Graph Attention Network)
- GATv2 (Graph Attention Network v2)
- GraphSAGE
- GCN (Graph Convolutional Network)

## Supported Datasets (Planetoid datasets)

- Cora
- CiteSeer
- PubMed 

## Usage

### Base Implementation

Navigate to the `base/` folder for standard GNN training:

```bash
cd base
python train.py --model GAT --dataset cora
```

### Cost-based Method

Navigate to the `cost/` folder for cost-based rejection:

```bash
cd cost
python train.py --model GraphSAGE --dataset citeseer --cost 0.7
```

### Coverage-based Method

Navigate to the `cov/` folder for coverage-based selective prediction:

```bash
cd cov
python train.py --model GCN --dataset pubmed --coverage 0.8
```

## Key Parameters

### Base Method
- `--model`: GNN architecture (GAT, GraphSAGE, GCN, GATv2)
- `--dataset`: Dataset name (cora, citeseer, pubmed)
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--hidden`: Number of hidden units
- `--nb_heads`: Number of attention heads (for GAT/GATv2)

### Cost-based Method
- `--cost`: Cost of rejection (default: 0.7)
- All base parameters apply

### Coverage-based Method  
- `--coverage`: Target coverage (default: 0.8)
- `--lamda`: Lambda parameter (default: 32)
- `--alphaloss`: Alpha for loss weighting (default: 0.5)
- All base parameters apply

## Acknowledgments

This code is inspired by and buildt upon GAT implementation in PyTorch: [pyGAT](https://github.com/Diego999/pyGAT)

## Citation

```bibtex
@article{kuchipudi2025node,
title={Node Classification With Reject Option},
author={Uday Bhaskar Kuchipudi and Jayadratha Gayen and Charu Sharma and Naresh Manwani},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=4xXJDO8Bvu},
note={}
}
```