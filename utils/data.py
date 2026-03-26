import os
from collections import defaultdict
from typing import List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config import CFG
from vocab import Vocabulary

def load_image_ids(split_file: str):
    ids = set()
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids

from collections import defaultdict

def build_splits_from_files(
    pairs,
    train_ids_file: str,
    val_ids_file: str,
    test_ids_file: str,
):
    train_ids = load_image_ids(train_ids_file)
    val_ids = load_image_ids(val_ids_file)
    test_ids = load_image_ids(test_ids_file)

    train, val, test = [], [], []

    for img_name, caption in pairs:
        if img_name in train_ids:
            train.append((img_name, caption))
        elif img_name in val_ids:
            val.append((img_name, caption))
        elif img_name in test_ids:
            test.append((img_name, caption))

    print(
        f"Official split | train images={len(train_ids)} val images={len(val_ids)} test images={len(test_ids)}"
    )
    print(
        f"Caption pairs | train={len(train)} val={len(val)} test={len(test)}"
    )

    return train, val, test

def load_caption_pairs(caption_file: str) -> List[Tuple[str, str]]:
    """
    Expected caption file format:
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs.
    """
    pairs = []
    with open(caption_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    start = 1 if lines and lines[0].lower().startswith("image") else 0
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        img_name, caption = parts
        pairs.append((img_name.strip(), caption.strip()))
    return pairs


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        image_dir: str,
        vocab: Vocabulary,
        transform=None,
        max_len: int = 30,
    ):
        self.samples = samples
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, caption = self.samples[idx]
        path = os.path.join(self.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption_ids = torch.tensor(
            self.vocab.numericalize(caption, self.max_len), dtype=torch.long
        )
        return image, caption_ids, caption, img_name


class CollateFn:
    def __init__(self, pad_idx: int, sort_by_length: bool = False):
        self.pad_idx = pad_idx
        self.sort_by_length = sort_by_length

    def __call__(self, batch):
        if self.sort_by_length:
            batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        images, cap_ids, raw_caps, img_names = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(x) for x in cap_ids], dtype=torch.long)
        captions = pad_sequence(cap_ids, batch_first=True, padding_value=self.pad_idx)
        return images, captions, lengths, raw_caps, img_names


def build_splits(
    pairs,
    seed: int = 42,
    train_split: float = None,
    val_split: float = None,
    use_official_split: bool = None,
    train_ids_file: str = None,
    val_ids_file: str = None,
    test_ids_file: str = None,
):
    if use_official_split is None:
        use_official_split = CFG.use_official_split

    if use_official_split:
        if train_ids_file is None:
            train_ids_file = CFG.train_ids_file
        if val_ids_file is None:
            val_ids_file = CFG.val_ids_file
        if test_ids_file is None:
            test_ids_file = CFG.test_ids_file

        return build_splits_from_files(
            pairs,
            train_ids_file=train_ids_file,
            val_ids_file=val_ids_file,
            test_ids_file=test_ids_file,
        )

    # fallback: random split
    if train_split is None:
        train_split = CFG.train_split
    if val_split is None:
        val_split = CFG.val_split

    img_to_caps = defaultdict(list)
    for img_name, caption in pairs:
        img_to_caps[img_name].append(caption)

    image_names = list(img_to_caps.keys())

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(image_names), generator=rng).tolist()
    image_names = [image_names[i] for i in perm]

    n = len(image_names)
    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_imgs = set(image_names[:n_train])
    val_imgs = set(image_names[n_train:n_train + n_val])
    test_imgs = set(image_names[n_train + n_val:])

    train, val, test = [], [], []

    for img_name, captions in img_to_caps.items():
        if img_name in train_imgs:
            train.extend([(img_name, c) for c in captions])
        elif img_name in val_imgs:
            val.extend([(img_name, c) for c in captions])
        else:
            test.extend([(img_name, c) for c in captions])

    print(
        f"Random split | train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}"
    )
    print(
        f"Caption pairs | train={len(train)} val={len(val)} test={len(test)}"
    )

    return train, val, test