import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from config import CFG
from utils.vocab import set_seed, Vocabulary
from utils.data import load_caption_pairs, build_splits, Flickr8kDataset, CollateFn
from models.baseline_model import ImageCaptioningModel
from engines.baseline_engine import show_predictions, evaluate_bleu_by_image


def main():
    set_seed(CFG.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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

    train_captions = [cap for _, cap in train_samples]

    vocab = Vocabulary(min_freq=CFG.min_word_freq)
    vocab.build(train_captions)
    print(f"Vocab size: {len(vocab)}")

    val_ds = Flickr8kDataset(val_samples, CFG.image_dir, vocab, transform, CFG.max_len)
    test_ds = Flickr8kDataset(test_samples, CFG.image_dir, vocab, transform, CFG.max_len)

    collate_fn = CollateFn(pad_idx=vocab.stoi["<pad>"])

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