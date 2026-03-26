import os

import torch

from config import CFG
from utils.utils import build_dataloaders, build_splits_and_vocab
from utils.vocab import set_seed
from models.baseline_model import ImageCaptioningModel
from engines.baseline_engine import show_predictions, evaluate_bleu_by_image


def main():
    set_seed(CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    train_samples, val_samples, test_samples, vocab = build_splits_and_vocab(CFG)
    _, val_ds, test_ds, _, val_loader, test_loader = build_dataloaders(
        CFG,
        vocab,
        train_samples,
        val_samples,
        test_samples,
    )

    model = ImageCaptioningModel(CFG, vocab_size=len(vocab), use_pretrained=False).to(CFG.device)

    ckpt_path = os.path.join(CFG.save_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model_state"])
    print("Loaded checkpoint for beam evaluation")

    beam_size = CFG.beam_size

    print("\nGreedy validation sample predictions:")
    show_predictions(
        model,
        val_loader,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        n=5,
        decode_method="greedy",
    )

    print("\nGreedy test sample predictions:")
    show_predictions(
        model,
        test_loader,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        n=5,
        decode_method="greedy",
    )

    val_bleu1_g, val_bleu4_g = evaluate_bleu_by_image(
        model,
        val_ds,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        limit_images=300,
        decode_method="greedy",
    )
    print(f"\nGreedy Validation BLEU-1: {val_bleu1_g:.4f} | BLEU-4: {val_bleu4_g:.4f}")

    test_bleu1_g, test_bleu4_g = evaluate_bleu_by_image(
        model,
        test_ds,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        limit_images=300,
        decode_method="greedy",
    )
    print(f"Greedy Test BLEU-1: {test_bleu1_g:.4f} | BLEU-4: {test_bleu4_g:.4f}")

    print(f"\nBeam validation sample predictions (beam_size={beam_size}):")
    show_predictions(
        model,
        val_loader,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        n=5,
        decode_method="beam",
        beam_size=beam_size,
    )

    print(f"\nBeam test sample predictions (beam_size={beam_size}):")
    show_predictions(
        model,
        test_loader,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        n=5,
        decode_method="beam",
        beam_size=beam_size,
    )

    val_bleu1_b, val_bleu4_b = evaluate_bleu_by_image(
        model,
        val_ds,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        limit_images=300,
        decode_method="beam",
        beam_size=beam_size,
    )
    print(f"\nBeam Validation BLEU-1: {val_bleu1_b:.4f} | BLEU-4: {val_bleu4_b:.4f}")

    test_bleu1_b, test_bleu4_b = evaluate_bleu_by_image(
        model,
        test_ds,
        vocab,
        CFG.device,
        max_len=CFG.max_len,
        limit_images=300,
        decode_method="beam",
        beam_size=beam_size,
    )
    print(f"Beam Test BLEU-1: {test_bleu1_b:.4f} | BLEU-4: {test_bleu4_b:.4f}")


if __name__ == "__main__":
    main()
