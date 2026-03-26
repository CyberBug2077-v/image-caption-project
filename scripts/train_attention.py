import os

import torch
import torch.nn as nn
import csv

from config import ATTN_CFG
from utils.utils import build_dataloaders, build_splits_and_vocab
from utils.vocab import set_seed
from models.attention_model import ImageCaptioningAttentionModel
from engines.attention_engine import (
    train_one_epoch_attention,
    validate_attention,
    show_predictions_attention,
    evaluate_bleu_by_image_attention,
)


def save_training_metrics_to_csv(history, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_bleu4"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(ATTN_CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.makedirs(ATTN_CFG.save_dir, exist_ok=True)

    train_samples, val_samples, test_samples, vocab = build_splits_and_vocab(ATTN_CFG)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        ATTN_CFG,
        vocab,
        train_samples,
        val_samples,
        test_samples,
        sort_by_length=True,
    )

    model = ImageCaptioningAttentionModel(ATTN_CFG, vocab_size=len(vocab)).to(ATTN_CFG.device)
    print("Built attention model")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=ATTN_CFG.lr)

    best_val = float("inf")
    best_val_bleu4 = float("-inf")
    eps = 1e-6
    history = []

    for epoch in range(1, ATTN_CFG.epochs + 1):
        train_loss, train_acc = train_one_epoch_attention(
            model, train_loader, optimizer, criterion, ATTN_CFG.device
        )
        val_loss, val_acc = validate_attention(
            model, val_loader, criterion, ATTN_CFG.device
        )
        _, val_bleu4 = evaluate_bleu_by_image_attention(
            model,
            val_ds,
            vocab,
            ATTN_CFG.device,
            max_len=ATTN_CFG.max_len,
            limit_images=300,
            decode_method="greedy",
            beam_size=ATTN_CFG.beam_size,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_bleu4={val_bleu4:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_bleu4": val_bleu4,
            }
        )

        should_save = False
        if val_bleu4 > best_val_bleu4 + eps:
            should_save = True
        elif abs(val_bleu4 - best_val_bleu4) <= eps and val_loss < best_val - eps:
            should_save = True

        if should_save:
            best_val = val_loss
            best_val_bleu4 = val_bleu4
            ckpt = {
                "model_state": model.state_dict(),
                "vocab_stoi": vocab.stoi,
                "config": ATTN_CFG.__dict__,
                "best_val_loss": best_val,
                "best_val_bleu4": best_val_bleu4,
                "selection_eps": eps,
            }
            torch.save(ckpt, os.path.join(ATTN_CFG.save_dir, "best_model.pt"))
            print(
                f"Saved best attention model (val_bleu4={best_val_bleu4:.4f}, "
                f"val_loss={best_val:.4f})"
            )

    metrics_csv_path = os.path.join("outputs", "logs", "attention_train_data.csv")
    save_training_metrics_to_csv(history, metrics_csv_path)
    print(f"Saved training metrics to {metrics_csv_path}")

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
        decode_method="greedy",
        beam_size=ATTN_CFG.beam_size,
    )

    print("\nTest sample predictions:")
    show_predictions_attention(
        model,
        test_loader,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        n=5,
        decode_method="greedy",
        beam_size=ATTN_CFG.beam_size,
    )

    val_bleu1, val_bleu4 = evaluate_bleu_by_image_attention(
        model,
        val_ds,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        limit_images=300,
        decode_method="greedy",
        beam_size=ATTN_CFG.beam_size,
    )
    print(f"\nValidation BLEU-1: {val_bleu1:.4f} | BLEU-4: {val_bleu4:.4f}")

    test_bleu1, test_bleu4 = evaluate_bleu_by_image_attention(
        model,
        test_ds,
        vocab,
        ATTN_CFG.device,
        max_len=ATTN_CFG.max_len,
        limit_images=300,
        decode_method="greedy",
        beam_size=ATTN_CFG.beam_size,
    )
    print(f"Test BLEU-1: {test_bleu1:.4f} | BLEU-4: {test_bleu4:.4f}")


if __name__ == "__main__":
    main()
