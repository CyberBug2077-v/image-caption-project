import os
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils.vocab import tokenize


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for images, captions, lengths, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        targets = pack_padded_sequence(
            captions[:, 1:],
            (lengths - 1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        ).data

        logits = model(images, captions, lengths)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (preds == targets).sum().item()
        total_tokens += targets.numel()

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for images, captions, lengths, _, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        targets = pack_padded_sequence(
            captions[:, 1:],
            (lengths - 1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        ).data

        logits = model(images, captions, lengths)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_correct += (preds == targets).sum().item()
        total_tokens += targets.numel()

    avg_loss = total_loss / max(len(loader), 1)
    avg_acc = total_correct / max(total_tokens, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def generate_captions(model, features, vocab, decode_method="greedy", beam_size=3, max_len=30):
    if decode_method == "greedy":
        return model.decoder.generate(features, vocab, max_len=max_len)

    if decode_method == "beam":
        preds = []
        for i in range(features.size(0)):
            pred = model.decoder.generate_beam(
                features[i:i+1],
                vocab,
                beam_size=beam_size,
                max_len=max_len,
            )
            preds.append(pred)
        return preds

    raise ValueError(f"Unknown decode_method: {decode_method}")


@torch.no_grad()
def show_predictions(model, loader, vocab, device, max_len: int = 30, n: int = 5, decode_method="greedy", beam_size=3):
    model.eval()
    images, captions, lengths, raw_caps, img_names = next(iter(loader))
    images = images.to(device)
    features = model.encoder(images)

    preds = generate_captions(
        model,
        features,
        vocab,
        decode_method=decode_method,
        beam_size=beam_size,
        max_len=max_len,
    )

    for i in range(min(n, len(preds))):
        print("-" * 60)
        print(f"Image: {img_names[i]}")
        print(f"Reference: {raw_caps[i]}")
        print(f"Prediction: {preds[i]}")


@torch.no_grad()
def evaluate_bleu_by_image(
    model,
    dataset,
    vocab,
    device,
    max_len: int = 30,
    limit_images=None,
    decode_method="greedy",
    beam_size=3,
):
    model.eval()

    img_to_refs = defaultdict(list)
    for img_name, caption in dataset.samples:
        img_to_refs[img_name].append(caption)

    image_names = list(img_to_refs.keys())
    if limit_images is not None:
        image_names = image_names[:limit_images]

    references = []
    hypotheses = []

    for img_name in tqdm(image_names, desc=f"BLEU-{decode_method}", leave=False):
        path = os.path.join(dataset.image_dir, img_name)
        image = Image.open(path).convert("RGB")

        if dataset.transform is not None:
            image = dataset.transform(image)

        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)

        pred_caption = generate_captions(
            model,
            features,
            vocab,
            decode_method=decode_method,
            beam_size=beam_size,
            max_len=max_len,
        )[0]

        pred_tokens = tokenize(pred_caption)
        if len(pred_tokens) == 0:
            pred_tokens = ["<unk>"]

        ref_tokens = [tokenize(c) for c in img_to_refs[img_name]]

        references.append(ref_tokens)
        hypotheses.append(pred_tokens)

    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(
        references,
        hypotheses,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth,
    )

    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth,
    )

    return bleu1, bleu4
