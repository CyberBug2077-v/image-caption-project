import csv
import os

import torch
import torch.nn as nn

from config import CFG
from utils.utils import build_dataloaders, build_splits_and_vocab
from utils.vocab import set_seed
from models.baseline_model import ImageCaptioningModel
from engines.baseline_engine import (
    evaluate_bleu_by_image,
    show_predictions,
    train_one_epoch,
    validate,
)


def save_training_metrics_to_csv(history, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_bleu4"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.makedirs(CFG.save_dir, exist_ok=True)

    train_samples, val_samples, test_samples, vocab = build_splits_and_vocab(
        CFG,
        load_existing_vocab=True,
        save_vocab=True,
    )
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        CFG,
        vocab,
        train_samples,
        val_samples,
        test_samples,
    )

    model = ImageCaptioningModel(CFG, vocab_size=len(vocab)).to(CFG.device)
    print("Built model")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

    best_val = float("inf")
    best_val_bleu4 = float("-inf")
    eps = 1e-6
    history = []

    for epoch in range(1, CFG.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CFG.device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, CFG.device)
        _, val_bleu4 = evaluate_bleu_by_image(
            model,
            val_ds,
            vocab,
            CFG.device,
            max_len=CFG.max_len,
            limit_images=300,
            decode_method="greedy",
            beam_size=CFG.beam_size,
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
                "config": CFG.__dict__,
                "best_val_loss": best_val,
                "best_val_bleu4": best_val_bleu4,
                "selection_eps": eps,
            }
            torch.save(ckpt, os.path.join(CFG.save_dir, "best_model.pt"))
            print(
                f"Saved best model (val_bleu4={best_val_bleu4:.4f}, "
                f"val_loss={best_val:.4f})"
            )

    metrics_csv_path = os.path.join("outputs", "logs", "baseline_train_data.csv")
    save_training_metrics_to_csv(history, metrics_csv_path)
    print(f"Saved training metrics to {metrics_csv_path}")

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
