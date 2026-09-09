"""DFL training with the paper's Gaussian perturbations -- Figures 5 and 11.

``--dp-mode exchange`` adds noise to local model states before exchange;
``--dp-mode aggregate`` adds noise to neighbourhood aggregates.
``--dp-mode none`` trains the reference model. Epsilon controls the noise
amplitude.

Writes the checkpoint triplet to assets/models_dp/ (tagged with the DP mode and
epsilon) and the per-epsilon test accuracy to results/dp_defense/.
Use --num-comm 2 to produce just the t0 / t0.5 / t1 states the attack needs.
The default 300 rounds match the length of the published accuracy logs.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from hssp_dfl import paths as _paths

MODEL_OUT = _paths.MODEL_DP_DIR
DATASET_OUT = _paths.DATASET_DIR
NETWORK_MAT = str(_paths.NETWORK_MAT)
MODEL_OUT.mkdir(parents=True, exist_ok=True)
DATASET_OUT.mkdir(parents=True, exist_ok=True)


import os
import argparse
import csv
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
from hssp_dfl.models import *

torch.set_default_dtype(torch.float64)

class args:
    batchsize = 500
    num_of_clients = 10
    num_comm =300
    IID = 1  # 1-iid  0-non-iid
    z_std = 0
    epo = 10
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


def _prepare_batch(img, label, device):
    if img.ndim == 3:
        if img.shape[0] in (1, 3) and img.shape[1] == img.shape[2]:
            img = img.unsqueeze(0)
        elif img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1).unsqueeze(0)
        else:
            img = img.unsqueeze(1)
    elif img.ndim == 4:
        if img.shape[1] in (1, 3):
            pass
        elif img.shape[-1] in (1, 3):
            img = img.permute(0, 3, 1, 2)
    label = label.long()
    return img.to(device), label.to(device)

# def add_dp_noise_to_state_dict(
#     state_dict,
#     epsilon,
#     delta=1e-5,
#     clipping_norm=0.5,
# ):
#     if epsilon <= 0:
#         return {name: tensor.clone() for name, tensor in state_dict.items()}

#     global_norm = torch.sqrt(sum(torch.sum(p**2) for p in state_dict.values() if torch.is_floating_point(p)))

#     clip_coef = clipping_norm / (global_norm + 1e-12)
#     if clip_coef < 1:
#         for name in state_dict:
#             if torch.is_floating_point(state_dict[name]):
#                 state_dict[name] *= clip_coef

#     sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
#     noise_std = sigma * clipping_norm

#     noisy_state = {}
#     for name, tensor in state_dict.items():
#         if torch.is_floating_point(tensor):
#             noise = torch.normal(
#                 mean=0.0,
#                 std=noise_std,
#                 size=tensor.shape,
#                 device=tensor.device,
#                 dtype=tensor.dtype,
#             )
#             noisy_state[name] = tensor + noise
#         else:
#             noisy_state[name] = tensor.clone()
            
#     return noisy_state

def add_dp_noise_to_state_dict(
    state_dict,
    epsilon,
    delta=1e-5,
    clipping_norm=1,
):
    """Reproduce the historical Gaussian-noise experiment without clipping.

    ``clipping_norm`` is a legacy name for the noise scale C in
    sigma = sqrt(2*log(1.25/delta))*C/epsilon.
    Input tensors are never modified in place.
    """
    if epsilon <= 0:
        return {name: tensor.clone() for name, tensor in state_dict.items()}

    sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise_std = sigma * clipping_norm

    noisy_state = {}
    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            noisy_state[name] = tensor + noise
        else:
            noisy_state[name] = tensor.clone()
            
    return noisy_state


def quantize_state_dict(state_dict, quantize):
    quantized_state = {}
    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            quantized_state[name] = torch.round(tensor * quantize) / quantize
        else:
            quantized_state[name] = tensor.clone()
    return quantized_state


def evaluate_model(model, data_loader, device):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for img, label in data_loader:
            img, label = _prepare_batch(img, label, device)
            output = model(img)
            loss = F.cross_entropy(output, label, reduction="sum")

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == label).sum().item()
            total_samples += label.size(0)

    if was_training:
        model.train()

    avg_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return avg_loss, accuracy


def save_metrics_csv(metrics_path, metric_rows):
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "dp_mode",
                "dp_epsilon",
                "train_loss_mean",
                "test_loss_mean",
                "test_acc_mean",
                "test_acc_std",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)


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
              "\nPass --force to overwrite, or set HSSP_MODEL_DP_DIR / HSSP_ASSETS to write elsewhere."
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-mode",
        choices=["none", "exchange", "aggregate"],
        default="none",
        help="Where to inject Gaussian DP noise: before exchange or after aggregation.",
    )
    parser.add_argument(
        "--dp-epsilon",
        type=float,
        default=0.0,
        help="Epsilon parameter controlling the Gaussian noise amplitude.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Batch size used for test accuracy evaluation.",
    )
    parser.add_argument(
        "--num-comm", "--num-communications",
        dest="num_comm",
        type=int,
        default=None,
        help=(
            "Override the number of communication rounds. The default keeps "
            "the reference accuracy logs' 300-round setting; use 2 when only t0/t0_5/t1 "
            "checkpoints are needed by the HSSP attack."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints instead of refusing.",
    )
    cli_args = parser.parse_args()

    args.dp_mode = cli_args.dp_mode
    args.dp_epsilon = cli_args.dp_epsilon
    args.eval_batch_size = cli_args.eval_batch_size
    if cli_args.num_comm is not None:
        if cli_args.num_comm < 1:
            parser.error("--num-comm must be at least 1")
        args.num_comm = cli_args.num_comm

    # setup_seed(20)
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _mode_tag = f"{args.dp_mode}_ep{str(args.dp_epsilon).replace('.', 'p')}"
    _model_tag = ("" if args.dp_mode == "none" and args.dp_epsilon == 0
                  else f"_{_mode_tag}")
    _guard(
        cli_args.force,
        MODEL_OUT / f"model_avg_ni{args.batchsize}_N{args.num_of_clients}_t0_z0_e{args.epo}{_model_tag}.pkl",
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

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    subset_length = args.num_of_clients * args.batchsize
    train_subset = Subset(training_dataset, range(subset_length))
    if args.IID == 1:
        shuffled_indices = torch.randperm(subset_length)
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

    with open(str(DATASET_OUT / 'dataset_ni{}_N{}.pkl'.format(args.batchsize, args.num_of_clients)), 'wb') as f:
        pickle.dump(split_datasets, f)

    if args.dataset=='cifar10' and args.model=='fc':
        model_init=fc(input_dim=32*32*3, output_dim=10)
    elif args.dataset=='mnist' and args.model=='cnn':
        model_init = cnn(output_dim=10)
    elif args.dataset=='cifar10' and args.model=='cnn':
        model_init = cnn_cifar(output_dim=10)

    mode_tag = f"{args.dp_mode}_ep{str(args.dp_epsilon).replace('.', 'p')}"
    model_tag = "" if args.dp_mode == "none" and args.dp_epsilon == 0 else f"_{mode_tag}"
    metrics_path = str(_paths.results_dir('dp_defense') / 'fl_dp_{}_{}_ni{}_N{}_e{}.csv'.format(
        args.dataset,
        args.model,
        args.batchsize,
        args.num_of_clients,
        args.epo,
    ))
    metrics_path = metrics_path.replace('.csv', f'{model_tag}.csv')

    nodes = []
    for j in range(args.num_of_clients):
        nodes.append(node(AA[j, :],
                          torch.utils.data.DataLoader(TensorDataset(split_datasets[j][0], split_datasets[j][1]),
                                                      batch_size=args.batchsize, shuffle=True), copy.deepcopy(model_init), device))

    metric_rows = []

    # Create/overwrite metrics file before training starts.
    save_metrics_csv(metrics_path, metric_rows)

    for k in range(args.num_comm):
        mod = []
        round_losses = []
        for i in range(args.num_of_clients):
            mod.append(nodes[i].model.state_dict())
        
        if k==0 or k==1:
            with open(str(MODEL_OUT / 'model_avg_ni{}_N{}_t{}_z{}_e{}{}.pkl'.format(
                    args.batchsize, args.num_of_clients, k, args.z_std,
                    args.epo, model_tag)), 'wb') as f:
                pickle.dump(mod, f)

        # sychronous
        for i in range(args.num_of_clients):
            loss = nodes[i].active_update(device, args.epo,args.quantize)
            round_losses.append(float(loss))

        mod = []
        for i in range(args.num_of_clients):
            current_state = copy.deepcopy(nodes[i].model.state_dict())
            if args.dp_mode == 'exchange' and args.dp_epsilon > 0:
                current_state = add_dp_noise_to_state_dict(current_state, args.dp_epsilon)
                current_state = quantize_state_dict(current_state, args.quantize)
                nodes[i].model.load_state_dict(current_state)
            mod.append(current_state)

        if k==0 or k==1:
            with open(str(MODEL_OUT / 'model_avg_ni{}_N{}_t{}_5_z{}_e{}{}.pkl'.format(
                    args.batchsize, args.num_of_clients, k, args.z_std,
                    args.epo, model_tag)), 'wb') as f:
                pickle.dump(mod, f)

        model_params = []
        for i in range(args.num_of_clients):
            model_params.append(copy.deepcopy(nodes[i].model.state_dict()))
        for i in range(args.num_of_clients):
            weighted_average_params = {}
            for param_name in model_params[0].keys():
                weighted_average_params[param_name] = torch.stack(
                    [model_params[j][param_name] * W[i][j]/scale_factor for j in range(len(model_params))]).sum(0)

            if args.dp_mode == 'aggregate' and args.dp_epsilon > 0:
                weighted_average_params = add_dp_noise_to_state_dict(weighted_average_params, args.dp_epsilon)
                weighted_average_params = quantize_state_dict(weighted_average_params, args.quantize)

            nodes[i].model.load_state_dict(weighted_average_params)

        test_accs = []
        test_losses = []
        for i in range(args.num_of_clients):
            test_loss, test_acc = evaluate_model(nodes[i].model, test_loader, device)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        round_train_loss = float(np.mean(round_losses)) if round_losses else 0.0
        round_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
        round_test_acc = float(np.mean(test_accs)) if test_accs else 0.0
        round_test_acc_std = float(np.std(test_accs)) if test_accs else 0.0

        metric_rows.append({
            "round": k,
            "dp_mode": args.dp_mode,
            "dp_epsilon": args.dp_epsilon,
            "train_loss_mean": round_train_loss,
            "test_loss_mean": round_test_loss,
            "test_acc_mean": round_test_acc,
            "test_acc_std": round_test_acc_std,
        })

        # Persist CSV every round so interrupted runs still keep progress.
        save_metrics_csv(metrics_path, metric_rows)

        print(
            f"round {k}: train_loss={round_train_loss:.6f}, "
            f"test_loss={round_test_loss:.6f}, test_acc={round_test_acc:.4f}"
        )

    print(f"Saved metrics -> {metrics_path}")
