import torch
import math
import numpy as np
from torch.utils.data import Dataset, Subset
from HParams import HParams

class ComboDataset(Dataset):
    def __init__(self, hp: HParams, device):
        self.hp = hp
        folder = hp.sketches_folder
        from os import listdir
        from os.path import isfile, join
        files = [f for f in listdir(folder) if isfile(join(folder, f))]

        all_data = []
        all_cats = []
        self.cat_labels = []
        for i in range(len(files)):
            self.cat_labels.append(files[i])
            file = join(folder, files[i])
            dataset = np.load(file, encoding='latin1')
            for ds_name in dataset:
                new_data = dataset[ds_name]
                new_data = purify(new_data, hp.max_seq_length)
                new_data = normalize(new_data)
                all_cats.append(np.full((len(new_data)), i))
                all_data.append(new_data)
            
        self.categories = np.concatenate(all_cats)
        self.data = np.concatenate(all_data)
        self.Nmax = max_size(self.data)
        self.device = device
        self.use_cache = False
        self.length = len(self.data)

        self.learned_cat = torch.stack([torch.tensor([-1], dtype=torch.int64, device=device)]*self.length)
        self.learned_style = torch.stack([torch.tensor([0.0,0.0], dtype=torch.float, device=device)]*self.length)

        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.use_cache:
            return self.cache_seqs[index], self.cache_lens[index], index
        else:
            seq, len = self.makeItem(index)
            return seq, len, index

    def CacheToSharedMemory(self):
        if self.hp.fast_debug:
            print("skipping cache because fast_debug is true")
            return

        print("loading data to cache")
        output_seq = []
        self.cache_lens = []
        for index in range(self.length):
            seq, seq_len = self.makeItem(index)
            output_seq.append(seq)
            self.cache_lens.append(seq_len)
        # self.cached_actions.share_memory_()
        # self.cached_states.share_memory_()
        self.cache_seqs = torch.stack(output_seq)
        self.use_cache = True
        del self.data
        self.data = None

    def makeItem(self, index):
        seq = self.data[index]

        len_seq = len(seq[:,0])
        new_seq = np.zeros((self.Nmax, 5))
        new_seq[:len_seq,:2] = seq[:,:2]
        new_seq[:len_seq-1,2] = 1-seq[:-1,2]
        new_seq[:len_seq,3] = seq[:,2]
        new_seq[(len_seq-1):,4] = 1
        new_seq[len_seq-1,2:4] = 0

        rtn = torch.as_tensor(new_seq, device=self.device, dtype=torch.float)

        return rtn, len_seq

    def NumCategoryTruths(self):
        return self.categories.max() + 1

    def GetCategoryTruths(self, indexes):
        return self.categories[indexes]

    def GetLearned(self, indexes):
        return self.learned_cat[indexes], self.learned_style[indexes]

    def SetLearned(self, indexes, categories, styles):
        self.learned_cat[indexes] = categories[:]
        self.learned_style[indexes, :] = styles[:, :]




def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes, max_seq_length):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if seq.shape[0] <= max_seq_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data