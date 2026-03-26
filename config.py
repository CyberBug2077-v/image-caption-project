from dataclasses import dataclass
import torch


@dataclass
class Config:
    image_dir: str = "./data/Flickr8k_images"
    caption_file: str = "./data/Flickr8k_text/captions.txt"
    train_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.trainImages.txt"
    val_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.devImages.txt"
    test_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.testImages.txt"
    vocab_path: str = "./outputs/vocab.pkl"
    use_official_split: bool = True

    # Only used when use_official_split = False
    train_split: float = 0.8
    val_split: float = 0.1

    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0

    min_word_freq: int = 5
    max_len: int = 30

    embed_dim: int = 256
    hidden_dim: int = 512
    feature_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.3

    lr: float = 1e-3
    epochs: int = 3
    debug_subset: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints"
    seed: int = 42
    beam_size: int = 3


CFG = Config()


@dataclass
class AttentionConfig:
    image_dir: str = "./data/Flickr8k_images"
    caption_file: str = "./data/Flickr8k_text/captions.txt"

    # official split files
    train_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.trainImages.txt"
    val_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.devImages.txt"
    test_ids_file: str = "./data/Flickr8k_text/split/Flickr_8k.testImages.txt"
    use_official_split: bool = True

    # only used if use_official_split = False
    train_split: float = 0.8
    val_split: float = 0.1

    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0

    min_word_freq: int = 5
    max_len: int = 30

    embed_dim: int = 256
    decoder_dim: int = 512
    encoder_dim: int = 512
    attention_dim: int = 256
    dropout: float = 0.3

    lr: float = 1e-3
    epochs: int = 5
    debug_subset: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints_attention"
    seed: int = 42
    use_pretrained_encoder: bool = True


ATTN_CFG = AttentionConfig()