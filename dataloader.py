from torch.utils.data import Dataset
import pickle


class TextIGNGraphDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, root_dir=None, name='train'):
        self.pickle_path = f'{root_dir}/{dataset}/{name}'

        with open(f"{self.pickle_path}/.meta", 'rb') as f:
            self.meta = pickle.load(f)

        self.chunk_length = self.meta['chunk_length']

        self.loaded_chunk_adj = None
        self.loaded_chunk_mask = None
        self.loaded_chunk_embed = None
        self.loaded_chunk_y = None

        self.loaded_index = -1

    def __len__(self):
        return self.meta['data_length']

    def __getitem__(self, idx):
        chunk = idx // self.chunk_length
        if chunk != self.loaded_index:
            with open(f"{self.pickle_path}/chunk.{chunk}.x_adj", 'rb') as f:
                self.loaded_chunk_adj = pickle.load(f)
            with open(f"{self.pickle_path}/chunk.{chunk}.x_adj_mask", 'rb') as f:
                self.loaded_chunk_mask = pickle.load(f)
            with open(f"{self.pickle_path}/chunk.{chunk}.x_embed", 'rb') as f:
                self.loaded_chunk_embed = pickle.load(f)
            with open(f"{self.pickle_path}/chunk.{chunk}.y", 'rb') as f:
                self.loaded_chunk_y = pickle.load(f)

            self.loaded_index = chunk

        return self.loaded_chunk_adj[idx % self.chunk_length], self.loaded_chunk_mask[idx % self.chunk_length], \
               self.loaded_chunk_embed[idx % self.chunk_length], self.loaded_chunk_y[idx % self.chunk_length]
