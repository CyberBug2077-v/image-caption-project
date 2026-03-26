import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from PIL import Image
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import ATTN_CFG, CFG
from engines.attention_engine import (
    evaluate_bleu_by_image_attention,
    generate_captions_attention,
    train_one_epoch_attention,
    validate_attention,
)
from engines.baseline_engine import (
    evaluate_bleu_by_image,
    generate_captions,
    train_one_epoch,
    validate,
)
from models.attention_model import ImageCaptioningAttentionModel
from models.baseline_model import ImageCaptioningModel
from utils.utils import build_dataloaders, build_splits_and_vocab
from utils.vocab import set_seed, tokenize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image captioning experiments with different model/decode combinations.",
    )
    parser.add_argument("--model", choices=["baseline", "attention"], required=True)
    parser.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--mode", choices=["train", "eval", "full"], default="full")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--bleu_limit", type=int, default=300)
    return parser.parse_args()


def ensure_output_dirs():
    for directory in [
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/results",
        "outputs/predictions",
        "outputs/attention_maps",
    ]:
        os.makedirs(directory, exist_ok=True)


def get_cfg(model_name, device, beam_size):
    cfg = copy.deepcopy(CFG if model_name == "baseline" else ATTN_CFG)
    cfg.device = device
    if hasattr(cfg, "beam_size"):
        cfg.beam_size = beam_size
    return cfg


def get_experiment_name(model_name, decode_method, beam_size):
    if decode_method == "beam":
        return f"{model_name}_beam{beam_size}"
    return f"{model_name}_greedy"


def get_paths(model_name, exp_name, override_checkpoint=None):
    checkpoint_path = override_checkpoint or os.path.join(
        "outputs",
        "checkpoints",
        f"{model_name}_best.pt",
    )
    log_path = os.path.join("outputs", "logs", f"{model_name}_train_metrics.csv")
    metrics_path = os.path.join("outputs", "results", f"{exp_name}_metrics.json")
    examples_path = os.path.join("outputs", "predictions", f"{exp_name}_examples.json")
    config_path = os.path.join("outputs", "results", f"{exp_name}_config.json")
    attention_maps_path = os.path.join("outputs", "attention_maps", f"{exp_name}_attention_maps.json")
    return checkpoint_path, log_path, metrics_path, examples_path, config_path, attention_maps_path


def save_history_to_csv(history, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_bleu4",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_model(model_name, cfg, vocab_size):
    if model_name == "baseline":
        return ImageCaptioningModel(cfg, vocab_size=vocab_size).to(cfg.device)
    return ImageCaptioningAttentionModel(cfg, vocab_size=vocab_size).to(cfg.device)


def prepare_data(model_name, cfg):
    train_samples, val_samples, test_samples, vocab = build_splits_and_vocab(
        cfg,
        load_existing_vocab=True,
        save_vocab=True,
    )
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        vocab,
        train_samples,
        val_samples,
        test_samples,
        sort_by_length=(model_name == "attention"),
    )
    return vocab, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_train_epoch(model_name, model, loader, optimizer, criterion, device):
    if model_name == "baseline":
        return train_one_epoch(model, loader, optimizer, criterion, device)
    return train_one_epoch_attention(model, loader, optimizer, criterion, device)


def run_val_epoch(model_name, model, loader, criterion, device):
    if model_name == "baseline":
        return validate(model, loader, criterion, device)
    return validate_attention(model, loader, criterion, device)


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    val_ds,
    vocab,
    cfg,
    checkpoint_path,
    log_path,
    decode_method,
    beam_size,
    bleu_limit,
    eps: float = 1e-6,
):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    best_val_bleu4 = float("-inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_train_epoch(
            model_name,
            model,
            train_loader,
            optimizer,
            criterion,
            cfg.device,
        )
        val_loss, val_acc = run_val_epoch(
            model_name,
            model,
            val_loader,
            criterion,
            cfg.device,
        )

        if model_name == "baseline":
            _, val_bleu4 = evaluate_bleu_by_image(
                model,
                val_ds,
                vocab,
                cfg.device,
                max_len=cfg.max_len,
                limit_images=bleu_limit,
                decode_method=decode_method,
                beam_size=beam_size,
            )
        else:
            _, val_bleu4 = evaluate_bleu_by_image_attention(
                model,
                val_ds,
                vocab,
                cfg.device,
                max_len=cfg.max_len,
                limit_images=bleu_limit,
                decode_method=decode_method,
                beam_size=beam_size,
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_bleu4": val_bleu4,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_bleu4={val_bleu4:.4f}"
        )

        should_save = False
        if val_bleu4 > best_val_bleu4 + eps:
            should_save = True
        elif abs(val_bleu4 - best_val_bleu4) <= eps and val_loss < best_val_loss - eps:
            should_save = True

        if should_save:
            best_val_bleu4 = val_bleu4
            best_val_loss = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "vocab_stoi": vocab.stoi,
                "vocab_itos": vocab.itos,
                "config": cfg.__dict__,
                "model_name": model_name,
                "best_val_loss": best_val_loss,
                "best_val_bleu4": best_val_bleu4,
                "selection_eps": eps,
            }
            torch.save(checkpoint, checkpoint_path)
            print(
                f"Saved best checkpoint to {checkpoint_path} "
                f"(val_bleu4={best_val_bleu4:.4f}, val_loss={best_val_loss:.4f})"
            )

    save_history_to_csv(history, log_path)
    print(f"Saved training log to {log_path}")
    return history


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def build_image_references(dataset, limit_images=None):
    img_to_refs = {}
    for img_name, caption in dataset.samples:
        if img_name not in img_to_refs:
            img_to_refs[img_name] = []
        img_to_refs[img_name].append(caption)

    image_names = list(img_to_refs.keys())
    if limit_images is not None:
        image_names = image_names[:limit_images]

    return image_names, img_to_refs


@torch.no_grad()
def generate_prediction_for_image(model_name, model, image_tensor, vocab, cfg, decode_method, beam_size):
    image_tensor = image_tensor.unsqueeze(0).to(cfg.device)

    if model_name == "baseline":
        features = model.encoder(image_tensor)
        prediction = generate_captions(
            model,
            features,
            vocab,
            decode_method=decode_method,
            beam_size=beam_size,
            max_len=cfg.max_len,
        )[0]
    else:
        prediction = generate_captions_attention(
            model,
            image_tensor,
            vocab,
            decode_method=decode_method,
            beam_size=beam_size,
            max_len=cfg.max_len,
        )[0]

    return prediction


@torch.no_grad()
def collect_predictions(model_name, model, dataset, vocab, cfg, decode_method, beam_size, limit_images=None):
    model.eval()
    image_names, img_to_refs = build_image_references(dataset, limit_images=limit_images)

    records = []
    for img_name in image_names:
        image = Image.open(os.path.join(dataset.image_dir, img_name)).convert("RGB")
        if dataset.transform is not None:
            image = dataset.transform(image)

        prediction = generate_prediction_for_image(
            model_name,
            model,
            image,
            vocab,
            cfg,
            decode_method,
            beam_size,
        )
        records.append(
            {
                "image": img_name,
                "references": img_to_refs[img_name],
                "prediction": prediction,
            }
        )
    return records


def collect_examples(records, num_examples):
    examples = []
    for record in records[:num_examples]:
        examples.append(
            {
                "image": record["image"],
                "reference": record["references"][0],
                "prediction": record["prediction"],
            }
        )
    return examples


def reshape_attention(alpha):
    if not alpha:
        return alpha

    side = int(math.sqrt(len(alpha)))
    if side * side == len(alpha):
        return [alpha[i * side:(i + 1) * side] for i in range(side)]
    return alpha


@torch.no_grad()
def build_attention_map_records(model, dataset, examples, vocab, cfg, decode_method, beam_size):
    attention_records = []

    for example in examples:
        image_path = os.path.join(dataset.image_dir, example["image"])
        image = Image.open(image_path).convert("RGB")
        if dataset.transform is not None:
            image = dataset.transform(image)

        image_tensor = image.unsqueeze(0).to(cfg.device)
        if decode_method == "beam":
            encoder_out = model.encoder(image_tensor)
            prediction, alphas = model.decoder.generate_beam(
                encoder_out,
                vocab,
                beam_size=beam_size,
                max_len=cfg.max_len,
                return_attention=True,
            )
        else:
            predictions, alphas_batch = model.generate(
                image_tensor,
                vocab,
                max_len=cfg.max_len,
                return_attention=True,
            )
            prediction = predictions[0]
            alphas = alphas_batch[0]

        words = prediction.split()
        attention_records.append(
            {
                "image_path": image_path,
                "prediction": prediction,
                "reference": example["reference"],
                "words": words,
                "alphas": [reshape_attention(alpha) for alpha in alphas[: len(words)]],
            }
        )

    return attention_records


def extract_ngrams(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def build_document_frequency(references_tokens):
    document_frequency = [defaultdict(int) for _ in range(4)]
    num_docs = 0

    for refs in references_tokens:
        for ref in refs:
            num_docs += 1
            for n in range(1, 5):
                for ngram in extract_ngrams(ref, n).keys():
                    document_frequency[n - 1][ngram] += 1

    return document_frequency, max(num_docs, 1)


def counts_to_tfidf(counts, doc_freq, num_docs):
    vec = {}
    for ngram, count in counts.items():
        df = doc_freq.get(ngram, 0)
        idf = math.log(num_docs / (1.0 + df))
        vec[ngram] = count * max(idf, 0.0)
    return vec


def cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_cider(references_tokens, hypotheses_tokens, sigma=6.0):
    doc_freqs, num_docs = build_document_frequency(references_tokens)
    scores = []

    for refs, hyp in zip(references_tokens, hypotheses_tokens):
        score_per_n = []
        for n in range(1, 5):
            hyp_vec = counts_to_tfidf(extract_ngrams(hyp, n), doc_freqs[n - 1], num_docs)
            sims = []
            for ref in refs:
                ref_vec = counts_to_tfidf(extract_ngrams(ref, n), doc_freqs[n - 1], num_docs)
                sim = cosine_similarity(hyp_vec, ref_vec)
                length_penalty = math.exp(-((len(hyp) - len(ref)) ** 2) / (2 * sigma * sigma))
                sims.append(sim * length_penalty)

            score_per_n.append(sum(sims) / max(len(sims), 1))

        scores.append(10.0 * sum(score_per_n) / 4.0)

    return sum(scores) / max(len(scores), 1)


def compute_text_metrics(records):
    references_tokens = []
    hypotheses_tokens = []

    for record in records:
        refs = [tokenize(caption) for caption in record["references"]]
        hyp = tokenize(record["prediction"])
        if not hyp:
            hyp = ["<unk>"]

        references_tokens.append(refs)
        hypotheses_tokens.append(hyp)

    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(
        references_tokens,
        hypotheses_tokens,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth,
    )
    bleu4 = corpus_bleu(
        references_tokens,
        hypotheses_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )
    meteor = sum(
        meteor_score(refs, hyp)
        for refs, hyp in zip(references_tokens, hypotheses_tokens)
    ) / max(len(hypotheses_tokens), 1)
    cider = compute_cider(references_tokens, hypotheses_tokens)
    avg_length = sum(len(hyp) for hyp in hypotheses_tokens) / max(len(hypotheses_tokens), 1)

    return {
        "bleu1": bleu1,
        "bleu4": bleu4,
        "meteor": meteor,
        "cider": cider,
        "avg_caption_length": avg_length,
    }


def evaluate_model(
    model_name,
    model,
    val_ds,
    test_ds,
    vocab,
    cfg,
    decode_method,
    beam_size,
    bleu_limit,
    num_examples,
):
    val_image_names, _ = build_image_references(val_ds, limit_images=None)
    test_image_names, _ = build_image_references(test_ds, limit_images=bleu_limit)

    val_records = collect_predictions(
        model_name,
        model,
        val_ds,
        vocab,
        cfg,
        decode_method,
        beam_size,
        limit_images=None,
    )
    test_records = collect_predictions(
        model_name,
        model,
        test_ds,
        vocab,
        cfg,
        decode_method,
        beam_size,
        limit_images=bleu_limit,
    )
    val_metrics = compute_text_metrics(val_records)
    test_metrics = compute_text_metrics(test_records)
    examples = collect_examples(test_records, num_examples)

    metrics = {
        "model": model_name,
        "decode": decode_method,
        "beam_size": beam_size if decode_method == "beam" else None,
        "val_bleu1": val_metrics["bleu1"],
        "val_bleu4": val_metrics["bleu4"],
        "val_meteor": val_metrics["meteor"],
        "val_cider": val_metrics["cider"],
        "val_avg_caption_length": val_metrics["avg_caption_length"],
        "test_bleu1": test_metrics["bleu1"],
        "test_bleu4": test_metrics["bleu4"],
        "test_meteor": test_metrics["meteor"],
        "test_cider": test_metrics["cider"],
        "test_avg_caption_length": test_metrics["avg_caption_length"],
        "num_examples": len(examples),
        "val_num_images_evaluated": len(val_image_names),
        "test_num_images_evaluated": len(test_image_names),
        "test_bleu_limit": bleu_limit,
    }
    return metrics, examples


def print_examples(examples):
    for item in examples:
        print("-" * 60)
        print(f"Image: {item['image']}")
        print(f"Reference: {item['reference']}")
        print(f"Prediction: {item['prediction']}")


def main():
    args = parse_args()
    ensure_output_dirs()

    exp_name = get_experiment_name(args.model, args.decode, args.beam_size)
    checkpoint_path, log_path, metrics_path, examples_path, config_path, attention_maps_path = get_paths(
        args.model,
        exp_name,
        override_checkpoint=args.checkpoint,
    )

    cfg = get_cfg(args.model, args.device, args.beam_size)
    set_seed(cfg.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"Running experiment: {exp_name}")
    print(f"Mode: {args.mode}")
    print(f"Device: {cfg.device}")

    save_json(
        {
            "args": vars(args),
            "cfg": cfg.__dict__,
            "experiment_name": exp_name,
        },
        config_path,
    )

    vocab, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = prepare_data(
        args.model,
        cfg,
    )
    model = build_model(args.model, cfg, len(vocab))
    print("Built model and dataloaders")

    if args.mode in {"train", "full"}:
        train_model(
            args.model,
            model,
            train_loader,
            val_loader,
            val_ds,
            vocab,
            cfg,
            checkpoint_path,
            log_path,
            args.decode,
            args.beam_size,
            args.bleu_limit,
        )

    if args.mode in {"eval", "full"}:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Run with --mode train/full first or pass --checkpoint."
            )

        load_checkpoint(model, checkpoint_path, cfg.device)
        print(f"Loaded checkpoint from {checkpoint_path}")

        metrics, examples = evaluate_model(
            args.model,
            model,
            val_ds,
            test_ds,
            vocab,
            cfg,
            args.decode,
            args.beam_size,
            args.bleu_limit,
            args.num_examples,
        )
        save_json(metrics, metrics_path)
        save_json(examples, examples_path)

        if args.model == "attention":
            attention_maps = build_attention_map_records(
                model,
                test_ds,
                examples,
                vocab,
                cfg,
                args.decode,
                args.beam_size,
            )
            save_json(attention_maps, attention_maps_path)

        print("Evaluation metrics:")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print("\nQualitative examples:")
        print_examples(examples)
        print(f"\nSaved metrics to {metrics_path}")
        print(f"Saved examples to {examples_path}")
        if args.model == "attention":
            print(f"Saved attention maps to {attention_maps_path}")


if __name__ == "__main__":
    main()
