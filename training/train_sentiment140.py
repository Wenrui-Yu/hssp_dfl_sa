"""DFL training on Sentiment140 -- produces the checkpoints for Figure 10 and Tables 2, 12-14.

Encodes tweets with text-embedding-ada-002 (cached in
assets/data/sentiment140/embeddings_ada002.pt; regenerating the cache needs an
OpenAI API key) and trains a logistic-regression classifier per node.

Writes assets/models/model_sentiment140_*.pkl,
assets/datasets/dataset_sentiment140_ni<B>_N<n>.pkl and the matching
texts_sentiment140_ni<B>_N<n>.pkl of ground-truth tweets.
"""

import argparse
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
import csv
import re
import numpy as np
from hssp_dfl.fedavg_node import node
import scipy.io as scio
import random
import math
from fractions import Fraction
from functools import reduce
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openai import OpenAI

import pickle
from hssp_dfl.models import *

torch.set_default_dtype(torch.float64)


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
              "\nPass --force to overwrite, or point HSSP_ASSETS somewhere else."
        )

_parser = argparse.ArgumentParser(
    description="DFL training on Sentiment140 (inputs for Figure 10 and Tables 2, 12-14).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--batch-size", type=int, default=1)
_parser.add_argument("--num-clients", type=int, default=10)
_parser.add_argument("--num-communications", type=int, default=2,
                     help="2 rounds produce the t0 / t0.5 / t1 triplet the attack needs")
_parser.add_argument("--local-epochs", type=int, default=1)
_parser.add_argument("--quantize", type=float, default=1e10)
_parser.add_argument("--max-samples", type=int, default=100,
                     help="tweets subsampled from the 1.6M corpus")
_parser.add_argument("--force", action="store_true",
                     help="overwrite existing checkpoints")
_CLI = _parser.parse_args()

class args:
    batchsize = _CLI.batch_size
    num_of_clients = _CLI.num_clients
    num_comm = _CLI.num_communications
    IID = 1
    z_std = 0
    epo = _CLI.local_epochs
    graph = 'load'  # line rgg load
    dataset = 'sentiment140'
    quantize = _CLI.quantize
    embedding_dim = 1536  # text-embedding-ada-002 dimension
    max_samples = _CLI.max_samples

_guard(
    _CLI.force,
    MODEL_OUT / f"model_sentiment140_ni{args.batchsize}_N{args.num_of_clients}_t0_z0_e{args.epo}.pkl",
    DATASET_OUT / f"dataset_sentiment140_ni{args.batchsize}_N{args.num_of_clients}.pkl",
)

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

def load_sentiment140(path, max_samples=100):
    """Load sentiment140 CSV and return texts and binary labels (0/1)."""
    texts = []
    labels = []
    with open(path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])  # 0=negative, 4=positive
            text = row[5]
            texts.append(text)
            labels.append(0 if label == 0 else 1)
    # Subsample for small-scale experiment
    indices = list(range(len(texts)))
    random.seed(0)
    random.shuffle(indices)
    indices = indices[:max_samples]
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    return texts, labels

if __name__ == '__main__':
    torch.manual_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    AA = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)
    W = np.zeros([args.num_of_clients, args.num_of_clients], dtype=int)

    scale_factor = 1

    if args.graph == 'rgg':
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
        denoms = [W_frac[i][j].denominator for i in range(args.num_of_clients) for j in range(args.num_of_clients) if isinstance(W_frac[i][j], Fraction)]
        scale_factor = int(reduce(math.lcm, denoms))
        for i in range(args.num_of_clients):
            for j in range(args.num_of_clients):
                W[i][j] = int(W_frac[i][j] * scale_factor)
    elif args.graph == 'line':
        for i in range(args.num_of_clients - 1):
            A[i][i + 1] = 1
            A[i + 1][i] = 1
            AA[i][i + 1] = 1
            AA[i + 1][i] = -1
        W[0][0] = 1
        W[args.num_of_clients - 1][args.num_of_clients - 1] = 1
        for i in range(args.num_of_clients - 1):
            W[i][i + 1] = 1
            W[i + 1][i] = 1
        scale_factor = 2
    elif args.graph == 'load':
        mat = scio.loadmat(NETWORK_MAT)
        A = mat['A']
        if 'AA' in mat:
            AA = mat['AA']
        else:
            AA = np.zeros_like(A)
            for i in range(args.num_of_clients):
                for j in range(i + 1, args.num_of_clients):
                    if A[i][j] == 1:
                        AA[i][j] = 1
                        AA[j][i] = -1
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

    diameter = graph_diameter(A)
    print('diameter:', diameter)

    # Load and preprocess sentiment140
    print('Loading sentiment140...')
    texts, labels = load_sentiment140(
        str(_paths.DATA_DIR / 'sentiment140' / 'training.1600000.processed.noemoticon.csv'),
        max_samples=args.max_samples
    )

    EMBED_CACHE = str(_paths.DATA_DIR / 'sentiment140' / 'embeddings_ada002.pt')
    if os.path.exists(EMBED_CACHE):
        print('Loading cached ada-002 embeddings...')
        features = torch.load(EMBED_CACHE).to(torch.float64)
    else:
        print('Computing ada-002 embeddings for {} texts via OpenAI API...'.format(len(texts)))
        client = OpenAI()
        EMBED_BATCH = 128
        all_embeddings = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            # Replace empty strings to avoid API errors
            batch = [t if len(t) > 0 else 'empty' for t in batch]
            response = client.embeddings.create(
                input=batch, model='text-embedding-ada-002', encoding_format='float'
            )
            emb = torch.tensor([e.embedding for e in response.data], dtype=torch.float64)
            all_embeddings.append(emb)
            if (i // EMBED_BATCH) % 10 == 0:
                print(f'  Embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}')
        features = torch.cat(all_embeddings, dim=0)
        torch.save(features, EMBED_CACHE)
        print(f'  Cached to {EMBED_CACHE}')

    labels = torch.tensor(labels, dtype=torch.long)
    input_dim = features.shape[1]
    print('Embedding dim:', input_dim)

    subset_length = args.num_of_clients * args.batchsize
    if args.IID == 1:
        shuffled_indices = torch.randperm(len(features))[:subset_length]
        training_inputs = features[shuffled_indices]
        training_labels = labels[shuffled_indices]
        selected_texts = [texts[i] for i in shuffled_indices.tolist()]
        split_datasets = list(
            zip(
                torch.split(training_inputs, args.batchsize),
                torch.split(training_labels.float(), args.batchsize)
            )
        )
        split_texts = [selected_texts[j * args.batchsize:(j + 1) * args.batchsize]
                       for j in range(args.num_of_clients)]
    else:
        sorted_indices = sorted(range(subset_length), key=lambda i: labels[i].item())
        num_shards = 2 * args.num_of_clients
        training_inputs = features[sorted_indices]
        training_labels = labels[sorted_indices]
        sorted_texts = [texts[i] for i in sorted_indices]
        shard_size = subset_length // num_shards
        shard_inputs = list(zip(torch.split(training_inputs, shard_size),
                                torch.split(training_labels.float(), shard_size)))
        shard_texts = [sorted_texts[s * shard_size:(s + 1) * shard_size] for s in range(num_shards)]
        index = list(range(num_shards))
        random.shuffle(index)
        split_datasets = []
        split_texts = []
        for i in range(args.num_of_clients):
            tmp0 = torch.cat((shard_inputs[index[i * 2]][0], shard_inputs[index[i * 2 + 1]][0]), 0)
            tmp1 = torch.cat((shard_inputs[index[i * 2]][1], shard_inputs[index[i * 2 + 1]][1]), 0)
            split_datasets.append((tmp0, tmp1))
            split_texts.append(shard_texts[index[i * 2]] + shard_texts[index[i * 2 + 1]])

    with open(str(DATASET_OUT / 'dataset_sentiment140_ni{}_N{}.pkl'.format(args.batchsize, args.num_of_clients)), 'wb') as f:
        pickle.dump(split_datasets, f)
    with open(str(DATASET_OUT / 'texts_sentiment140_ni{}_N{}.pkl'.format(args.batchsize, args.num_of_clients)), 'wb') as f:
        pickle.dump(split_texts, f)

    model_init = logistic_regression(input_dim=input_dim, output_dim=2)

    # Sentiment140 data is tabular TF-IDF vectors, override active_update to skip image reshaping
    class sentiment_node(node):
        def active_update(self, device, epo=1, quantize=1e10):
            for e in range(epo):
                img, label = next(iter(self.dataloader))
                label = label.type(torch.LongTensor)
                img, label = img.to(device), label.to(device)
                self.optimizer.zero_grad()
                output = self.model(img)
                loss = F.cross_entropy(output, label)
                loss.backward()
                self.optimizer.step()

            quantized_state_dict = {}
            for param_name, param_tensor in self.model.state_dict().items():
                quantized_state_dict[param_name] = torch.round(param_tensor * quantize) / quantize
            self.model.load_state_dict(quantized_state_dict)
            return loss.item()

    nodes = []
    for j in range(args.num_of_clients):
        nodes.append(sentiment_node(AA[j, :],
                          torch.utils.data.DataLoader(TensorDataset(split_datasets[j][0], split_datasets[j][1]),
                                                      batch_size=args.batchsize, shuffle=True),
                          copy.deepcopy(model_init), device))

    for k in range(args.num_comm):
        mod = []
        for i in range(args.num_of_clients):
            mod.append(nodes[i].model.state_dict())
        with open(str(MODEL_OUT / 'model_sentiment140_ni{}_N{}_t{}_z{}_e{}.pkl'.format(args.batchsize, args.num_of_clients, k,
                                                                        args.z_std, args.epo)), 'wb') as f:
            pickle.dump(mod, f)

        # synchronous
        for i in range(args.num_of_clients):
            loss = nodes[i].active_update(device, args.epo, args.quantize)

        mod = []
        for i in range(args.num_of_clients):
            mod.append(nodes[i].model.state_dict())
        with open(str(MODEL_OUT / 'model_sentiment140_ni{}_N{}_t{}_5_z{}_e{}.pkl'.format(args.batchsize, args.num_of_clients, k,
                                                                          args.z_std, args.epo)), 'wb') as f:
            pickle.dump(mod, f)

        model_params = []
        for i in range(args.num_of_clients):
            model_params.append(copy.deepcopy(nodes[i].model.state_dict()))
        for i in range(args.num_of_clients):
            weighted_average_params = {}
            for param_name in model_params[0].keys():
                weighted_average_params[param_name] = torch.stack(
                    [model_params[j][param_name] * W[i][j] / scale_factor for j in range(len(model_params))]).sum(0)
            nodes[i].model.load_state_dict(weighted_average_params)
