from utils import *
import numpy as np
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Creating pickles for dataloader')
parser.add_argument('--dataset', default='mr', help='Training dataset')  # 'mr','ohsumed','R8','R52'
parser.add_argument('--pickle_length', default=10000, type=int, help='Number of epochs to train.')
args = parser.parse_args()


# Prepare train dataset
def create_pickle(dataset, name, names, pickle_length):
    adj, emd, y = load_data(dataset, names)

    names = ['x_adj', 'x_embed', 'y']

    if not os.path.exists(f'dataloader/{dataset}/{name}'):
        Path(f'dataloader/{dataset}/{name}').mkdir(parents=True, exist_ok=True)

    adj, mask = preprocess_adj(adj)
    emd = preprocess_features(emd)

    pickle_count = len(adj) // pickle_length + (1 if len(adj) % pickle_length else 0)

    meta = {'dataset_type': name, 'dataset': dataset, 'data_length': len(adj), 'chunk_length': pickle_length,
            'chunk_count': pickle_count}

    with open(f"dataloader/{dataset}/{name}/.meta", 'wb') as f:
        pkl.dump(meta, f)

    for i in range(0, len(adj), pickle_length):
        with open(f"dataloader/{dataset}/{name}/chunk.{i // pickle_length}.{names[0]}", 'wb') as f:
            pkl.dump(adj[i:i + pickle_length], f)

        with open(f"dataloader/{dataset}/{name}/chunk.{i // pickle_length}.{names[0]}_mask", 'wb') as f:
            pkl.dump(mask[i:i + pickle_length], f)

        with open(f"dataloader/{dataset}/{name}/chunk.{i // pickle_length}.{names[1]}", 'wb') as f:
            pkl.dump(emd[i:i + pickle_length], f)

        with open(f"dataloader/{dataset}/{name}/chunk.{i // pickle_length}.{names[2]}", 'wb') as f:
            pkl.dump(y[i:i + pickle_length], f)


create_pickle(dataset=args.dataset, name='train', names=['x_adj', 'x_embed', 'y'], pickle_length=args.pickle_length)
create_pickle(dataset=args.dataset, name='test', names=['tx_adj', 'tx_embed', 'ty'], pickle_length=args.pickle_length)
# create_pickle(dataset=args.dataset, name='all', names=['allx_adj', 'allx_embed', 'ally'],
              # pickle_length=args.pickle_length)
