import os
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from utils.vocab import tokenize


def train_one_epoch_attention(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, captions, lengths, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        scores, decode_lengths, alphas = model(images, captions, lengths)
        targets = captions[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_attention(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, captions, lengths, _, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        captions = captions.to(device)

        scores, decode_lengths, alphas = model(images, captions, lengths)
        targets = captions[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def show_predictions_attention(model, loader, vocab, device, max_len: int = 30, n: int = 5):
    model.eval()
    images, captions, lengths, raw_caps, img_names = next(iter(loader))
    images = images.to(device)
    preds = model.generate(images, vocab, max_len=max_len)

    for i in range(min(n, len(preds))):
        print("-" * 60)
        print(f"Image: {img_names[i]}")
        print(f"Reference: {raw_caps[i]}")
        print(f"Prediction: {preds[i]}")


@torch.no_grad()
def evaluate_bleu_by_image_attention(
    model,
    dataset,
    vocab,
    device,
    max_len: int = 30,
    limit_images=None,
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

    for img_name in tqdm(image_names, desc="BLEU-attention", leave=False):
        path = os.path.join(dataset.image_dir, img_name)
        image = Image.open(path).convert("RGB")

        if dataset.transform is not None:
            image = dataset.transform(image)

        image = image.unsqueeze(0).to(device)
        pred_caption = model.generate(image, vocab, max_len=max_len)[0]

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