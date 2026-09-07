"""DFL training on CIFAR-10 -- produces the checkpoints the attack consumes.

Writes, per communication round k:

    assets/models/model_avg_ni<B>_N<n>_t<k>_z0_e<E>.pkl     state before local step
    assets/models/model_avg_ni<B>_N<n>_t<k>_5_z0_e<E>.pkl   local update (t = k + 1/2)
    assets/datasets/dataset_ni<B>_N<n>.pkl                  the per-node data split

The attack needs the t0 / t0.5 / t1 triplet, i.e. --num-communications 2.
The topology is read from assets/network.mat with --graph load.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from hssp_dfl import paths as _paths

MODEL_OUT = _paths.MODEL_DIR
DATASET_OUT = _paths.DATASET_DIR
NETWORK_MAT = str(_paths.NETWORK_MAT)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
DATASET_OUT.mkdir(parents=True, exist_ok=True)


import os
import argparse
import numpy as np
from hssp_dfl.fedavg_node import node
import scipy.io as scio
import random
import math
from fractions import Fraction
from functools import reduce
import copy

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
from torch.utils.data import Subset, Dataset

import collections
import pickle
from pathlib import Path
from hssp_dfl.models import *

torch.set_default_dtype(torch.float64)

class args:
    batchsize = 1
    num_of_clients = 10
    num_comm =2
    IID = 1  # 1-iid  0-non-iid
    z_std = 0
    epo = 1
    graph='load' #line rgg
    dataset='cifar10' #mnist cifar10
    model='cnn' #fc cnn
    quantize=1e10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def floyd_warshall(adj_matrix):
    num_vertices = adj_matrix.shape[0]
    dist = adj_matrix.astype(float)
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    return dist

def graph_diameter(adj_matrix):
    dist_matrix = floyd_warshall(adj_matrix)
    diameter = np.max(dist_matrix[dist_matrix != np.inf])
    return diameter

def flatten_params(state_dict):
    params = []
    for key, param in state_dict.items():
        params.append(param.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train CIFAR/MNIST DFL snapshots for reconstruction experiments."
    )
    parser.add_argument("--batch-size", type=int, default=args.batchsize)
    parser.add_argument("--num-clients", type=int, default=args.num_of_clients)
    parser.add_argument("--num-communications", type=int, default=args.num_comm)
    parser.add_argument("--local-epochs", type=int, default=args.epo)
    parser.add_argument("--iid", type=int, choices=(0, 1), default=args.IID)
    parser.add_argument("--noise-std", type=float, default=args.z_std)
    parser.add_argument("--graph", choices=("line", "rgg", "load"), default=args.graph)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default=args.dataset)
    parser.add_argument("--model", choices=("fc", "cnn"), default=args.model)
    parser.add_argument("--quantize", type=float, default=args.quantize)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for dataset and model snapshot pickles (default: assets/).",
    )
    parser.add_argument(
        "--gradient-output-dir",
        type=Path,
        help="Optionally save the exact pre-optimizer gradient for every node and round.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints instead of refusing.",
    )
    parser.add_argument(
        "--nested-batch-max-size",
        type=int,
        help=(
            "Use node-wise nested IID batches drawn from a fixed pool of this size. "
            "This keeps model initialization and B1 subset B2 subset ... comparisons controlled."
        ),
    )
    return parser

def apply_cli_args(cli_args):
    if cli_args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if cli_args.num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if cli_args.num_communications <= 0:
        raise ValueError("--num-communications must be positive")
    if cli_args.local_epochs <= 0:
        raise ValueError("--local-epochs must be positive")
    if cli_args.quantize <= 0:
        raise ValueError("--quantize must be positive")
    if cli_args.nested_batch_max_size is not None:
        if cli_args.nested_batch_max_size < cli_args.batch_size:
            raise ValueError("--nested-batch-max-size must be at least --batch-size")
        if cli_args.iid != 1:
            raise ValueError("--nested-batch-max-size currently requires --iid 1")

    args.batchsize = cli_args.batch_size
    args.num_of_clients = cli_args.num_clients
    args.num_comm = cli_args.num_communications
    args.epo = cli_args.local_epochs
    args.IID = cli_args.iid
    args.z_std = cli_args.noise_std
    args.graph = cli_args.graph
    args.dataset = cli_args.dataset
    args.model = cli_args.model
    args.quantize = cli_args.quantize


def _guard(force, *targets):
    """Refuse to overwrite existing checkpoints unless --force is given.

    ``assets/`` is often a set of symlinks into a shared checkpoint store, so a
    stray run must not silently replace the exact files the paper's numbers were
    produced on.
    """
    existing = [str(t) for t in targets if _Path(t).exists()]
    if existing and not force:
        raise SystemExit(
            "refusing to overwrite existing checkpoints:\n  "
            + "\n  ".join(existing)
            + "\n\nThese may be the exact files the paper's results were produced on"
              "\n(assets/ is frequently a symlink into a shared store)."
              "\nPass --force to overwrite, or --output-dir / HSSP_ASSETS to write elsewhere."
        )


if __name__ == '__main__':
    cli_args = build_parser().parse_args()
    apply_cli_args(cli_args)

    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_visible_devices
    setup_seed(cli_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (cli_args.output_dir or _paths.ASSETS).resolve()
    model_output_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    _guard(
        cli_args.force,
        model_output_dir / f"model_avg_ni{args.batchsize}_N{args.num_of_clients}_t0_z0_e{args.epo}.pkl",
        output_dir / "datasets" / f"dataset_ni{args.batchsize}_N{args.num_of_clients}.pkl",
    )

    A = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    AA = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    W = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)

    scale_factor = 1

    if args.graph=='rgg':
        # decentralized network - RGG
        radius = np.sqrt(np.log(args.num_of_clients) / args.num_of_clients)
        np.random.seed(5)
        location = np.reshape(np.random.random(2 * args.num_of_clients), (args.num_of_clients, 2))
        for i in range(args.num_of_clients):
            for j in range(i + 1, args.num_of_clients):
                if np.sum((location[i] - location[j]) ** 2) <= radius ** 2:
                    A[i][j] = 1
                    A[j][i] = 1
                    AA[i][j] = 1
                    AA[j][i] = -1
        deg = np.sum(A, axis=0)
        W_frac = np.full((args.num_of_clients, args.num_of_clients), Fraction(0), dtype=object)
        for i in range(args.num_of_clients):
            for j in range(args.num_of_clients):
                if i != j and A[i][j] == 1:
                    W_frac[i][j] = Fraction(1, 1 + int(max(deg[i], deg[j])))
        for i in range(args.num_of_clients):
            W_frac[i][i] = Fraction(1) - sum(W_frac[i][j] for j in range(args.num_of_clients) if i != j)
        # Compute scale_factor as LCM of all denominators
        denoms = [W_frac[i][j].denominator for i in range(args.num_of_clients) for j in range(args.num_of_clients) if isinstance(W_frac[i][j], Fraction)]
        scale_factor = int(reduce(math.lcm, denoms))
        for i in range(args.num_of_clients):
            for j in range(args.num_of_clients):
                W[i][j] = int(W_frac[i][j] * scale_factor)
    elif args.graph=='line':
        # decentralized network - line graph
        for i in range(args.num_of_clients-1):
            A[i][i+1] = 1
            A[i+1][i] = 1
            AA[i][i+1] = 1
            AA[i+1][i] = -1
        W[0][0] = 1
        W[args.num_of_clients - 1][args.num_of_clients - 1] = 1
        for i in range(args.num_of_clients - 1):
            W[i][i + 1] = 1
            W[i + 1][i] = 1
        scale_factor = 2
    elif args.graph=='load':
        # load pre-defined decentralized network
        mat = scio.loadmat(NETWORK_MAT)
        A = mat['A']
        AA = mat['AA']
        W = np.array(mat['W'], dtype=int)
        scale_factor = int(mat['scale_factor'].flat[0])
        honest_nodes = mat['honest_nodes'].flatten().tolist()
        corrupt_nodes = mat['corrupt_nodes'].flatten().tolist()
        sg_honest = mat['sg_honest'].flatten().tolist()
        sg_corrupt = mat['sg_corrupt'].flatten().tolist()
        print(f'Loaded network: honest={honest_nodes}, corrupt={corrupt_nodes}')
        print(f'  Subgraph honest={sg_honest}, corrupt={sg_corrupt}')

    P = A[:]
    tmp = A[:]
    for i in range(1, args.num_of_clients):
        tmp = np.dot(tmp, A)
        P = P + tmp
    assert np.count_nonzero(P) == args.num_of_clients ** 2

    diameter=graph_diameter(A)
    print('diameter:',diameter)

    if args.dataset=='cifar10':
        training_dataset = torchvision.datasets.CIFAR10(
            root=str(_paths.DATA_DIR / "cifar10"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(_paths.DATA_DIR / "cifar10"),
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        )
    elif args.dataset=='mnist':
        training_dataset = torchvision.datasets.MNIST(
            root=str(_paths.DATA_DIR / "mnist"),
            train=True,
            download=True,
            transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(_paths.DATA_DIR / "mnist"),
            train=False,
            download=True,
            transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        )

    subset_length = args.num_of_clients * args.batchsize
    train_subset = Subset(training_dataset, range(subset_length))
    if args.IID == 1:
        if cli_args.nested_batch_max_size is None:
            shuffled_indices = torch.randperm(subset_length)
        else:
            pool_size = args.num_of_clients * cli_args.nested_batch_max_size
            if pool_size > len(training_dataset):
                raise ValueError(
                    f"nested batch pool requests {pool_size} examples from "
                    f"a dataset of size {len(training_dataset)}"
                )
            data_generator = torch.Generator().manual_seed(cli_args.seed)
            pool_permutation = torch.randperm(pool_size, generator=data_generator)
            shuffled_indices = torch.cat(
                [
                    pool_permutation[
                        node_index * cli_args.nested_batch_max_size:
                        (node_index + 1) * cli_args.nested_batch_max_size
                    ][:args.batchsize]
                    for node_index in range(args.num_of_clients)
                ]
            )
        # training_inputs = training_dataset.train_data[shuffled_indices]
        training_inputs = training_dataset.data[shuffled_indices]
        training_inputs = training_inputs / 128 - 1
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]
        # training_labels = torch.Tensor(training_dataset.train_labels)[shuffled_indices]
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), args.batchsize),
                torch.split(torch.Tensor(training_labels), args.batchsize)
            )
        )
    else:
        sorted_indices = sorted(range(subset_length), key=lambda i: train_subset[i][1])
        num_shards = 2 * args.num_of_clients
        # training_inputs = training_dataset.data[sorted_indices]
        training_inputs = training_dataset.train_data[sorted_indices]
        training_inputs = training_inputs / 128 - 1
        # training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]
        training_labels = torch.Tensor(training_dataset.train_labels)[sorted_indices]
        shard_size = args.num_of_clients * args.batchsize // num_shards
        shard_inputs = list(zip(torch.split(torch.Tensor(training_inputs), shard_size),
                                torch.split(torch.Tensor(training_labels), shard_size)))
        index = list(range(num_shards))
        random.shuffle(index)
        split_datasets = []
        for i in range(args.num_of_clients):
            tmp0 = torch.cat((shard_inputs[index[i * 2]][0], shard_inputs[index[i * 2 + 1]][0]), 0)
            tmp1 = torch.cat((shard_inputs[index[i * 2]][1], shard_inputs[index[i * 2 + 1]][1]), 0)
            split_datasets.append((tmp0, tmp1))

    (output_dir / 'datasets').mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / 'datasets' / 'dataset_ni{}_N{}.pkl'.format(
        args.batchsize,
        args.num_of_clients,
    )
    with dataset_path.open('wb') as f:
        pickle.dump(split_datasets, f)

    if args.dataset=='cifar10' and args.model=='fc':
        model_init=fc(input_dim=32*32*3, output_dim=10)
    elif args.dataset=='mnist' and args.model=='cnn':
        model_init = cnn(output_dim=10)
    elif args.dataset=='cifar10' and args.model=='cnn':
        model_init = cnn_cifar(output_dim=10)

    nodes = []
    for j in range(args.num_of_clients):
        nodes.append(node(AA[j, :],
                          torch.utils.data.DataLoader(TensorDataset(split_datasets[j][0], split_datasets[j][1]),
                                                      batch_size=args.batchsize, shuffle=True), copy.deepcopy(model_init), device))

    if cli_args.gradient_output_dir is not None:
        cli_args.gradient_output_dir.mkdir(parents=True, exist_ok=True)

    for k in range(args.num_comm):
        mod = []
        for i in range(args.num_of_clients):
            mod.append(nodes[i].model.state_dict())
        before_path = model_output_dir / 'model_avg_ni{}_N{}_t{}_z{}_e{}.pkl'.format(
            args.batchsize, args.num_of_clients, k, args.z_std, args.epo
        )
        with before_path.open('wb') as f:
            pickle.dump(mod, f)

        # sychronous
        true_gradients_by_node = {}
        losses_by_node = {}
        for i in range(args.num_of_clients):
            if cli_args.gradient_output_dir is None:
                loss = nodes[i].active_update(device, args.epo,args.quantize)
            else:
                loss, true_gradients = nodes[i].active_update(
                    device,
                    args.epo,
                    args.quantize,
                    capture_gradients=True,
                )
                true_gradients_by_node[int(i)] = true_gradients
                losses_by_node[int(i)] = float(loss)

        if cli_args.gradient_output_dir is not None:
            gradient_path = cli_args.gradient_output_dir / (
                "true_gradients_ni{}_N{}_t{}_z{}_e{}.pkl".format(
                    args.batchsize,
                    args.num_of_clients,
                    k,
                    args.z_std,
                    args.epo,
                )
            )
            with gradient_path.open("wb") as gradient_handle:
                pickle.dump(
                    {
                        "format": "dfl-true-gradients-v1",
                        "gradients_by_node": true_gradients_by_node,
                        "metadata": {
                            "batch_size": args.batchsize,
                            "communication_round": k,
                            "local_epochs": args.epo,
                            "learning_rate": float(nodes[0].lr),
                            "num_clients": args.num_of_clients,
                            "seed": cli_args.seed,
                            "nested_batch_max_size": cli_args.nested_batch_max_size,
                            "losses_by_node": losses_by_node,
                            "gradient_timing": "after backward and before optimizer.step",
                        },
                    },
                    gradient_handle,
                )
            print(f"Saved exact node gradients: {gradient_path}")

        mod = []
        for i in range(args.num_of_clients):
            mod.append(nodes[i].model.state_dict())
        local_update_path = model_output_dir / 'model_avg_ni{}_N{}_t{}_5_z{}_e{}.pkl'.format(
            args.batchsize, args.num_of_clients, k, args.z_std, args.epo
        )
        with local_update_path.open('wb') as f:
            pickle.dump(mod, f)

        model_params = []
        for i in range(args.num_of_clients):
            model_params.append(copy.deepcopy(nodes[i].model.state_dict()))
        for i in range(args.num_of_clients):
            weighted_average_params = {}
            for param_name in model_params[0].keys():
                weighted_average_params[param_name] = torch.stack(
                    [model_params[j][param_name] * W[i][j]/scale_factor for j in range(len(model_params))]).sum(0)
            nodes[i].model.load_state_dict(weighted_average_params)

