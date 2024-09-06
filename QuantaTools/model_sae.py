import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import transformer_lens.utils as utils
import itertools
from sklearn.model_selection import ParameterGrid
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig


def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))


def safe_kl_div(p, q, eps=1e-8):
    p = torch.as_tensor(p)
    q = torch.as_tensor(q)
    
    p = torch.clamp(p, eps, 1-eps)
    q = torch.clamp(q, eps, 1-eps)
    
    return p * (torch.log(p) - torch.log(q)) + (1-p) * (torch.log(1-p) - torch.log(1-q))


class AdaptiveSparseAutoencoder(nn.Module):
    def __init__(self, encoding_dim, input_dim, sparsity_target=0.05, sparsity_weight=1e-3, l1_weight=1e-5):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.l1_weight = l1_weight  # L1 regularization weight to promote sparsity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.ReLU()
        ).cuda()
        self.decoder = nn.Linear(self.encoding_dim, input_dim).cuda()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def loss(self, x, encoded, decoded):
        mse_loss = nn.MSELoss()(decoded, x)
        
        # KL divergence for sparsity
        avg_activation = torch.mean(encoded, dim=0)
        kl_div = safe_kl_div(self.sparsity_target, avg_activation)
        sparsity_penalty = torch.sum(kl_div)
        
        # L1 regularization on activations
        l1_penalty = torch.mean(torch.abs(encoded))
        
        # Clip sparsity penalty to avoid exploding gradients
        sparsity_penalty = torch.clamp(sparsity_penalty, max=1e3)
        
        total_loss = mse_loss + self.sparsity_weight * sparsity_penalty + self.l1_weight * l1_penalty
        
        if not torch.isfinite(total_loss):
            print(f"Warning: Non-finite loss detected.")
            print(f"MSE: {mse_loss.item()}")
            print(f"Sparsity Penalty: {sparsity_penalty.item()}")
            print(f"L1 Penalty: {l1_penalty.item()}")
            print(f"Avg Activation stats: min={avg_activation.min().item()}, max={avg_activation.max().item()}, mean={avg_activation.mean().item()}")
            print(f"Encoded stats: min={encoded.min().item()}, max={encoded.max().item()}, mean={encoded.mean().item()}")
            total_loss = torch.clamp(total_loss, max=1e6)  # Clamp to a large but finite value
        
        return total_loss, mse_loss.item(), sparsity_penalty.item(), l1_penalty.item()


class SparseAutoencoderConfig(PretrainedConfig):
    model_type = "sparse_autoencoder"

    def __init__(
        self,
        encoding_dim=128,
        input_dim=768,
        sparsity_target=0.05,
        sparsity_weight=1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight


class SparseAutoencoderForHF(PreTrainedModel):
    config_class = SparseAutoencoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.sae = AdaptiveSparseAutoencoder(
            config.encoding_dim,
            config.input_dim,
            config.sparsity_target,
            config.sparsity_weight
        )

    def forward(self, x):
        return self.sae(x)

    def save_pretrained(self, save_directory, **kwargs):
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, f"{save_directory}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin")
        model.load_state_dict(state_dict)
        return model


def save_sae_to_huggingface(sae, save_directory):
    config = SparseAutoencoderConfig(
        encoding_dim=sae.encoding_dim,
        input_dim=sae.input_dim,
        sparsity_target=sae.sparsity_target,
        sparsity_weight=sae.sparsity_weight
    )
    model_for_hf = SparseAutoencoderForHF(config)
    model_for_hf.sae = sae
    model_for_hf.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")
