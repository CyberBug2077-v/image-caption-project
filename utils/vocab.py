import re
from collections import Counter
from typing import List
import random
import numpy as np
import torch
import pickle
from pathlib import Path


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


class Vocabulary:
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.stoi = dict(SPECIAL_TOKENS)
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    def build(self, captions: List[str]) -> None:
        counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))

        idx = len(self.stoi)
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "stoi": self.stoi,
                "itos": self.itos,
                "min_freq": self.min_freq
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.stoi = data["stoi"]
        self.itos = data["itos"]
        self.min_freq = data["min_freq"]

    def numericalize(self, text: str, max_len: int) -> List[int]:
        tokens = tokenize(text)[: max_len - 2]
        ids = [self.stoi["<sos>"]]
        ids += [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]
        ids += [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            token = self.itos.get(int(idx), "<unk>")
            if token in {"<sos>", "<pad>"}:
                continue
            if token == "<eos>":
                break
            words.append(token)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.stoi)
    