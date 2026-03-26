import os
from typing import Optional, Sequence, Tuple

from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.data import CollateFn, Flickr8kDataset, build_splits, load_caption_pairs
from utils.vocab import SPECIAL_TOKENS, Vocabulary


def build_transform(cfg_or_image_size):
    image_size = getattr(cfg_or_image_size, "image_size", cfg_or_image_size)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_vocab(
    cfg,
    train_samples: Sequence[Tuple[str, str]],
    load_existing_vocab: bool = False,
    save_vocab: bool = False,
):
    vocab = Vocabulary(min_freq=cfg.min_word_freq)
    vocab_path = getattr(cfg, "vocab_path", None)

    if load_existing_vocab and vocab_path and os.path.exists(vocab_path):
        print("Loading existing vocab...")
        vocab.load(vocab_path)
        if len(vocab) > len(SPECIAL_TOKENS):
            return vocab

        print("Existing vocab is degenerate; rebuilding from train data...")

    print("Building vocab from train data...")
    train_captions = [caption for _, caption in train_samples]
    vocab.build(train_captions)

    if save_vocab and vocab_path:
        vocab.save(vocab_path)

    return vocab


def build_splits_and_vocab(cfg, load_existing_vocab: bool = False, save_vocab: bool = False):
    pairs = load_caption_pairs(cfg.caption_file)
    print(f"Loaded {len(pairs)} image-caption pairs")

    train_samples, val_samples, test_samples = build_splits(
        pairs,
        seed=cfg.seed,
        train_split=getattr(cfg, "train_split", None),
        val_split=getattr(cfg, "val_split", None),
        use_official_split=getattr(cfg, "use_official_split", None),
        train_ids_file=getattr(cfg, "train_ids_file", None),
        val_ids_file=getattr(cfg, "val_ids_file", None),
        test_ids_file=getattr(cfg, "test_ids_file", None),
    )

    if getattr(cfg, "debug_subset", 0):
        train_samples = train_samples[: cfg.debug_subset]
        val_samples = val_samples[: min(1000, len(val_samples))]
        test_samples = test_samples[: min(1000, len(test_samples))]
        print(
            f"Debug subset enabled | train={len(train_samples)} "
            f"val={len(val_samples)} test={len(test_samples)}"
        )

    vocab = build_vocab(
        cfg,
        train_samples,
        load_existing_vocab=load_existing_vocab,
        save_vocab=save_vocab,
    )
    print(f"Vocab size: {len(vocab)}")

    return train_samples, val_samples, test_samples, vocab


def build_datasets(
    cfg,
    vocab,
    train_samples,
    val_samples,
    test_samples,
    transform: Optional[T.Compose] = None,
):
    if transform is None:
        transform = build_transform(cfg)

    train_ds = Flickr8kDataset(
        train_samples,
        cfg.image_dir,
        vocab,
        transform,
        cfg.max_len,
    )
    val_ds = Flickr8kDataset(
        val_samples,
        cfg.image_dir,
        vocab,
        transform,
        cfg.max_len,
    )
    test_ds = Flickr8kDataset(
        test_samples,
        cfg.image_dir,
        vocab,
        transform,
        cfg.max_len,
    )

    return train_ds, val_ds, test_ds


def build_dataloaders(
    cfg,
    vocab,
    train_samples,
    val_samples,
    test_samples,
    sort_by_length: bool = False,
    transform: Optional[T.Compose] = None,
):
    train_ds, val_ds, test_ds = build_datasets(
        cfg,
        vocab,
        train_samples,
        val_samples,
        test_samples,
        transform=transform,
    )
    print("Built datasets")

    collate_fn = CollateFn(
        pad_idx=vocab.stoi["<pad>"],
        sort_by_length=sort_by_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    print("Built loaders")

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
