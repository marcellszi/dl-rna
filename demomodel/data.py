import demomodel.config as config
import demomodel.utils as utils

import numpy as np
import pathlib
import torch

# One-hot encoding dictionary for IUPAC symbols
# See: https://www.bioinformatics.org/sms/iupac.html
IUPAC_ONEHOT = {
    #     A  C  G  U
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "U": [0, 0, 0, 1],
    "R": [1, 0, 1, 0],
    "Y": [0, 1, 0, 1],
    "S": [0, 1, 1, 0],
    "W": [1, 0, 0, 1],
    "K": [0, 0, 1, 1],
    "M": [1, 1, 0, 0],
    "B": [0, 1, 1, 1],
    "D": [1, 0, 1, 1],
    "H": [1, 1, 0, 1],
    "V": [1, 1, 1, 0],
    "N": [1, 1, 1, 1],
    ".": [0, 0, 0, 0],
    "-": [0, 0, 0, 0],
}
IUPAC_MAPPING = {b: np.where(np.array(s) == 1) for b, s in IUPAC_ONEHOT.items()}


class CTData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.cts = list(pathlib.Path(path).glob("*.ct"))

    def __len__(self):
        return len(self.cts)

    def _encode(self, path, number=0):
        x = np.zeros((4, config.max_length), dtype=np.float32)
        y = np.zeros((config.max_length), dtype=np.float32)
        metadata = {}
        metadata["name"] = path.stem

        title, bases, pairings = utils.read_ct(path, number=number)
        metadata["title"], metadata["length"] = title, len(bases)

        for i, (b, p) in enumerate(zip(bases, pairings)):
            x[IUPAC_MAPPING[b], i - 1] = 1
            if p != -1:
                y[i - 1] = 1

        return metadata, torch.tensor(x), torch.tensor(y)

    def __getitem__(self, idx):
        return self._encode(self.cts[idx])


class SEQData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.seqs = list(pathlib.Path(path).glob("*.seq"))

    def __len__(self):
        return len(self.seqs)

    def _encode(self, path):
        x = np.zeros((4, config.max_length), dtype=np.float32)
        metadata = {}
        metadata["name"] = path.stem

        title, sequence = utils.read_seq(path)
        metadata["title"] = len(sequence)
        metadata["length"] = len(sequence)
        for i, b in enumerate(sequence):
            x[IUPAC_MAPPING[b], i - 1] = 1

        return metadata, torch.tensor(x), torch.tensor(0)

    def __getitem__(self, idx):
        return self._encode(self.seqs[idx])
