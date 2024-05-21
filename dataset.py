import torch
from torch.utils.data import Dataset
import numpy as np

class Shakespeare(Dataset):
    """ Shakespeare dataset """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.data = f.read()

        self.chars = sorted(set(self.data))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        self.data_indices = [self.char_to_idx[ch] for ch in self.data]

        self.seq_length = 30
        self.data_len = len(self.data_indices) - self.seq_length

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        input_seq = self.data_indices[idx:idx+self.seq_length]
        target_seq = self.data_indices[idx+1:idx+self.seq_length+1]

        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        
        return input_seq, target_seq

if __name__ == '__main__':
    dataset = Shakespeare('shakespeare_train.txt')
    print(f"Dataset length: {len(dataset)}")
    print(f"First sample input: {dataset[0][0]}")
    print(f"First sample target: {dataset[0][1]}")
