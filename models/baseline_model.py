import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import Config
from utils.vocab import Vocabulary


class EncoderCNN(nn.Module):
    def __init__(self, feature_dim: int = 512, use_pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(backbone.fc.in_features, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images).flatten(1)
        feats = self.bn(self.proj(feats))
        return feats


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        feature_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions[:, :-1])
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)

        from torch.nn.utils.rnn import pack_padded_sequence

        packed = pack_padded_sequence(
            embeddings,
            (lengths - 1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed, (h0, c0))
        outputs = self.fc(self.dropout(packed_out.data))
        return outputs

    @torch.no_grad()
    def generate(self, features, vocab: Vocabulary, max_len: int = 30):
        batch_size = features.size(0)
        device = features.device
        h = self.init_h(features).unsqueeze(0)
        c = self.init_c(features).unsqueeze(0)

        inputs = torch.full(
            (batch_size, 1), vocab.stoi["<sos>"], dtype=torch.long, device=device
        )
        generated = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            emb = self.embedding(inputs[:, -1:])
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.fc(out.squeeze(1))
            next_token = logits.argmax(dim=-1)
            inputs = torch.cat([inputs, next_token.unsqueeze(1)], dim=1)

            for i in range(batch_size):
                generated[i].append(int(next_token[i]))

        return [vocab.decode(seq) for seq in generated]

    @torch.no_grad()
    def generate_beam(self, features, vocab: Vocabulary, beam_size: int = 3, max_len: int = 30):
        assert features.size(0) == 1, "generate_beam only supports batch size 1"

        device = features.device
        sos_id = vocab.stoi["<sos>"]
        eos_id = vocab.stoi["<eos>"]

        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)

        beams = [([sos_id], 0.0, h0, c0)]
        completed = []

        def rank_key(seq, score):
            return score / max(1, len(seq) - 1)

        for _ in range(max_len):
            candidates = []

            for seq, score, h, c in beams:
                last_token = seq[-1]

                if last_token == eos_id:
                    completed.append((seq, score))
                    candidates.append((seq, score, h, c))
                    continue

                inp = torch.tensor([[last_token]], dtype=torch.long, device=device)
                emb = self.embedding(inp)
                out, (h_new, c_new) = self.lstm(emb, (h, c))
                logits = self.fc(out.squeeze(1))
                log_probs = F.log_softmax(logits, dim=-1)

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for k in range(beam_size):
                    token_id = int(topk_ids[0, k].item())
                    token_score = float(topk_log_probs[0, k].item())

                    new_seq = seq + [token_id]
                    new_score = score + token_score
                    candidates.append((new_seq, new_score, h_new, c_new))

            candidates = sorted(
                candidates,
                key=lambda x: rank_key(x[0], x[1]),
                reverse=True
            )[:beam_size]

            beams = candidates

            if all(seq[-1] == eos_id for seq, _, _, _ in beams):
                break

        if completed:
            best_seq, _ = max(completed, key=lambda x: rank_key(x[0], x[1]))
        else:
            best_seq, _, _, _ = max(beams, key=lambda x: rank_key(x[0], x[1]))

        return vocab.decode(best_seq)


class ImageCaptioningModel(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int, use_pretrained: bool = True):
        super().__init__()
        self.encoder = EncoderCNN(cfg.feature_dim, use_pretrained=use_pretrained)
        self.decoder = DecoderLSTM(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            feature_dim=cfg.feature_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs