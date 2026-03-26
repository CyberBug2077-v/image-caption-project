import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config import ATTN_CFG
from utils.vocab import set_seed, Vocabulary
from utils.data import load_caption_pairs, build_splits, Flickr8kDataset, CollateFn
from models.attention_model import ImageCaptioningAttentionModel
from engines.attention_engine import (
    train_one_epoch_attention,
    validate_attention,
    show_predictions_attention,
    evaluate_bleu_by_image_attention,
)


def main():
    set_seed(ATTN_CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.makedirs(ATTN_CFG.save_dir, exist_ok=True)

    transform = T.Compose([
        T.Resize((ATTN_CFG.image_size, ATTN_CFG.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pairs = load_caption_pairs(ATTN_CFG.caption_file)
    print(f"Loaded {len(pairs)} image-caption pairs")

    train_samples, val_samples, test_samples = build_splits(
        pairs,
        seed=ATTN_CFG.seed,
        train_split=ATTN_CFG.train_split,
        val_split=ATTN_CFG.val_split,
    )

    if ATTN_CFG.debug_subset:
        train_samples = train_samples[: ATTN_CFG.debug_subset]
        val_samples = val_samples[: min(1000, len(val_samples))]
        test_samples = test_samples[: min(1000, len(test_samples))]
        print(
            f"Debug subset enabled | train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
        )

    train_captions = [cap for _, cap in train_samples]

    vocab = Vocabulary(min_freq=ATTN_CFG.min_word_freq)
    vocab.build(train_captions)
    print(f"Vocab size: {len(vocab)}")

    train_ds = Flickr8kDataset(
        train_samples, ATTN_CFG.image_dir, vocab, transform, ATTN_CFG.max_len
    )
    val_ds = Flickr8kDataset(
        val_samples, ATTN_CFG.image_dir, vocab, transform, ATTN_CFG.max_len
    )
    test_ds = Flickr8kDataset(
        test_samples, ATTN_CFG.image_dir, vocab, transform, ATTN_CFG.max_len
    )
    print("Built datasets")

    collate_fn = CollateFn(
        pad_idx=vocab.stoi["<pad>"],
        sort_by_length=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ATTN_CFG.batch_size,
        shuffle=True,
        num_workers=ATTN_CFG.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ATTN_CFG.batch_size,
        shuffle=False,
        num_workers=ATTN_CFG.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=ATTN_CFG.batch_size,
        shuffle=False,
        num_workers=ATTN_CFG.num_workers,
        collate_fn=collate_fn,
    )
    print("Built loaders")

    model = ImageCaptioningAttentionModel(ATTN_CFG, vocab_size=len(vocab)).to(ATTN_CFG.device)
    print("Built attention model")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=ATTN_CFG.lr)

    best_val = float("inf")

    for epoch in range(1, ATTN_CFG.epochs + 1):
        train_loss = train_one_epoch_attention(
            model, train_loader, optimizer, criterion, ATTN_CFG.device
        )
        val_loss = validate_attention(
            model, val_loader, criterion, ATTN_CFG.device
        )
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "vocab_stoi": vocab.stoi,
                "config": ATTN_CFG.__dict__,
            }
            torch.save(ckpt, os.path.join(ATTN_CFG.save_dir, "best_model.pt"))
            print("Saved best attention model")

    best_ckpt_path = os.path.join(ATTN_CFG.save_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt_path, map_location=ATTN_CFG.device)
    model.load_state_dict(ckpt["model_state"])
    print("Loaded best attention model for final evaluation")

    print("\nValidation sample predictions:")
    show_predictions_attention(
        model,
        val_loader,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        n=5,
    )

    print("\nTest sample predictions:")
    show_predictions_attention(
        model,
        test_loader,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        n=5,
    )

    val_bleu1, val_bleu4 = evaluate_bleu_by_image_attention(
        model,
        val_ds,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        limit_images=300,
    )
    print(f"\nValidation BLEU-1: {val_bleu1:.4f} | BLEU-4: {val_bleu4:.4f}")

    test_bleu1, test_bleu4 = evaluate_bleu_by_image_attention(
        model,
        test_ds,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        limit_images=300,
    )
    print(f"Test BLEU-1: {test_bleu1:.4f} | BLEU-4: {test_bleu4:.4f}")


if __name__ == "__main__":
    main()