import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config import CFG
from utils.vocab import set_seed, Vocabulary
from utils.data import load_caption_pairs, build_splits, Flickr8kDataset, CollateFn
from models.baseline_model import ImageCaptioningModel
from engines.baseline_engine import train_one_epoch, validate, show_predictions


def main():
    set_seed(CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.makedirs(CFG.save_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((CFG.image_size, CFG.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pairs = load_caption_pairs(CFG.caption_file)
    print(f"Loaded {len(pairs)} image-caption pairs")

    train_samples, val_samples, test_samples = build_splits(pairs, CFG.seed)

    if CFG.debug_subset:
        train_samples = train_samples[: CFG.debug_subset]
        val_samples = val_samples[: min(1000, len(val_samples))]
        test_samples = test_samples[: min(1000, len(test_samples))]
        print(
            f"Debug subset enabled | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
        )

    train_csv_path = os.path.join("outputs", "logs", "baselline_train_data.csv")
    save_train_samples_to_csv(train_samples, train_csv_path)

    train_captions = [cap for _, cap in train_samples]

    vocab = Vocabulary(min_freq=CFG.min_word_freq)
    if os.path.exists(CFG.vocab_path):
        print("Loading existing vocab...")
        vocab.load(CFG.vocab_path)
    else:
        print("Building vocab from train data...")
        train_captions = [c for _, c in train_samples]
        vocab.build(train_captions)
        vocab.save(CFG.vocab_path)

    print(f"Vocab size: {len(vocab)}")

    train_ds = Flickr8kDataset(train_samples, CFG.image_dir, vocab, transform, CFG.max_len)
    val_ds = Flickr8kDataset(val_samples, CFG.image_dir, vocab, transform, CFG.max_len)
    test_ds = Flickr8kDataset(test_samples, CFG.image_dir, vocab, transform, CFG.max_len)
    print("Built datasets")

    collate_fn = CollateFn(pad_idx=vocab.stoi["<pad>"])

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
    )
    print("Built loaders")

    model = ImageCaptioningModel(CFG, vocab_size=len(vocab)).to(CFG.device)
    print("Built model")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_val = float("inf")

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss = validate(model, val_loader, criterion, CFG.device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "vocab_stoi": vocab.stoi,
                "config": CFG.__dict__,
            }
            torch.save(ckpt, os.path.join(CFG.save_dir, "best_model.pt"))
            print("Saved best model")

    best_ckpt_path = os.path.join(CFG.save_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model_state"])
    print("Loaded best model for final evaluation")

    print("\nValidation sample predictions:")
    show_predictions(model, val_loader, vocab, CFG.device, max_len=CFG.max_len, n=5)

    print("\nTest sample predictions:")
    show_predictions(model, test_loader, vocab, CFG.device, max_len=CFG.max_len, n=5)


if __name__ == "__main__":
    main()
