import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import AttentionConfig


class EncoderCNNAttention(nn.Module):
    def __init__(self, encoder_dim: int = 512, use_pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # B, 512, 7, 7

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.encoder_dim = encoder_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(images)                       # B, 512, 7, 7
        feats = feats.permute(0, 2, 3, 1)                  # B, 7, 7, 512
        feats = feats.view(feats.size(0), -1, feats.size(-1))  # B, 49, 512
        return feats


class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        att1 = self.encoder_att(encoder_out)                     # B, num_pixels, attention_dim
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)    # B, 1, attention_dim
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # B, num_pixels
        alpha = self.softmax(att)                               # B, num_pixels
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size

        embeddings = self.embedding(captions[:, :-1])
        decode_lengths = (lengths - 1).tolist()
        max_decode_len = max(decode_lengths)

        predictions = torch.zeros(
            batch_size, max_decode_len, vocab_size, device=encoder_out.device
        )
        alphas = torch.zeros(
            batch_size, max_decode_len, encoder_out.size(1), device=encoder_out.device
        )

        h, c = self.init_hidden_state(encoder_out)

        for t in range(max_decode_len):
            batch_size_t = sum(l > t for l in decode_lengths)

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h_t, c_t = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )

            h = torch.cat([h_t, h[batch_size_t:]], dim=0)
            c = torch.cat([c_t, c[batch_size_t:]], dim=0)

            preds = self.fc(self.dropout(h_t))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, decode_lengths, alphas

    @torch.no_grad()
    def generate(
        self,
        encoder_out: torch.Tensor,
        vocab,
        max_len: int = 30,
        return_attention: bool = False,
    ):
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        eos_id = vocab.stoi["<eos>"]

        h, c = self.init_hidden_state(encoder_out)
        prev_words = torch.full(
            (batch_size,),
            vocab.stoi["<sos>"],
            dtype=torch.long,
            device=device,
        )

        generated = [[] for _ in range(batch_size)]
        attention_maps = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            active = ~finished
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=False).squeeze(1)
            embeddings = self.embedding(prev_words[active_idx])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[active_idx],
                h[active_idx],
            )

            gate = self.sigmoid(self.f_beta(h[active_idx]))
            attention_weighted_encoding = gate * attention_weighted_encoding

            h_new, c_new = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h[active_idx], c[active_idx]),
            )

            scores = self.fc(h_new)
            next_words = scores.argmax(dim=1)
            h[active_idx] = h_new
            c[active_idx] = c_new
            prev_words[active_idx] = next_words

            for local_idx, (batch_idx, token) in enumerate(zip(active_idx.tolist(), next_words.tolist())):
                if token == eos_id:
                    finished[batch_idx] = True
                    continue

                generated[batch_idx].append(int(token))
                attention_maps[batch_idx].append(alpha[local_idx].detach().cpu().tolist())

        captions = [vocab.decode(seq) for seq in generated]
        if return_attention:
            return captions, attention_maps
        return captions

    @torch.no_grad()
    def generate_beam(
        self,
        encoder_out: torch.Tensor,
        vocab,
        beam_size: int = 3,
        max_len: int = 30,
        return_attention: bool = False,
    ):
        assert encoder_out.size(0) == 1, "generate_beam only supports batch size 1"

        device = encoder_out.device
        sos_id = vocab.stoi["<sos>"]
        eos_id = vocab.stoi["<eos>"]

        h0, c0 = self.init_hidden_state(encoder_out)
        beams = [([sos_id], 0.0, h0, c0, [])]
        completed = []

        def rank_key(seq, score):
            return score / max(1, len(seq) - 1)

        for _ in range(max_len):
            candidates = []

            for seq, score, h, c, alpha_history in beams:
                last_token = seq[-1]

                if last_token == eos_id:
                    completed.append((seq, score, alpha_history))
                    candidates.append((seq, score, h, c, alpha_history))
                    continue

                prev_words = torch.tensor([last_token], dtype=torch.long, device=device)
                embeddings = self.embedding(prev_words)
                attention_weighted_encoding, alpha = self.attention(encoder_out, h)

                gate = self.sigmoid(self.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                h_new, c_new = self.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1),
                    (h, c),
                )

                logits = self.fc(h_new)
                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

                for k in range(beam_size):
                    token_id = int(topk_ids[0, k].item())
                    token_score = float(topk_log_probs[0, k].item())
                    new_seq = seq + [token_id]
                    new_score = score + token_score
                    new_alpha_history = list(alpha_history)
                    if token_id != eos_id:
                        new_alpha_history.append(alpha[0].detach().cpu().tolist())
                    candidates.append((new_seq, new_score, h_new, c_new, new_alpha_history))

            beams = sorted(
                candidates,
                key=lambda x: rank_key(x[0], x[1]),
                reverse=True,
            )[:beam_size]

            if all(seq[-1] == eos_id for seq, _, _, _, _ in beams):
                break

        if completed:
            best_seq, _, best_alphas = max(completed, key=lambda x: rank_key(x[0], x[1]))
        else:
            best_seq, _, _, _, best_alphas = max(beams, key=lambda x: rank_key(x[0], x[1]))

        caption = vocab.decode(best_seq)
        if return_attention:
            return caption, best_alphas
        return caption


class ImageCaptioningAttentionModel(nn.Module):
    def __init__(self, cfg: AttentionConfig, vocab_size: int):
        super().__init__()
        self.encoder = EncoderCNNAttention(
            encoder_dim=cfg.encoder_dim,
            use_pretrained=cfg.use_pretrained_encoder,
        )
        self.decoder = DecoderWithAttention(
            attention_dim=cfg.attention_dim,
            embed_dim=cfg.embed_dim,
            decoder_dim=cfg.decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=cfg.encoder_dim,
            dropout=cfg.dropout,
        )

    def forward(self, images, captions, lengths):
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions, lengths)

    @torch.no_grad()
    def generate(self, images, vocab, max_len: int = 30, return_attention: bool = False):
        encoder_out = self.encoder(images)
        return self.decoder.generate(
            encoder_out,
            vocab,
            max_len=max_len,
            return_attention=return_attention,
        )
