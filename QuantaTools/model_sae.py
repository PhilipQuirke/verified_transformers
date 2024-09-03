import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import transformer_lens.utils as utils

class AdaptiveSparseAutoencoder(nn.Module):
    def __init__(self, encoding_dim, sparsity_weight=1e-5):
        super().__init__()
        self.encoding_dim = encoding_dim
        self.sparsity_weight = sparsity_weight
        self.encoder = None
        self.decoder = None
    
    def initialize(self, input_dim):
        self.encoder = nn.Linear(input_dim, self.encoding_dim).cuda()
        self.decoder = nn.Linear(self.encoding_dim, input_dim).cuda()
    
    def forward(self, x):
        if self.encoder is None or self.decoder is None:
            self.initialize(x.shape[-1])
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def loss(self, x, encoded, decoded):
        mse_loss = nn.MSELoss()(decoded, x)
        sparsity_loss = torch.mean(torch.abs(encoded))
        return mse_loss + self.sparsity_weight * sparsity_loss

def train_sae(sae, activation_generator, batch_size=64, num_epochs=100, learning_rate=1e-3):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for activations in activation_generator:
            dataloader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)
            
            for batch in dataloader:
                x = batch[0].cuda()
                x.requires_grad_(True)
                
                optimizer.zero_grad()
                encoded, decoded = sae(x)
                loss = sae.loss(x, encoded, decoded)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/num_batches:.4f}")

def analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=32, chunk_size=100):
    def extract_mlp_activations_in_chunks(model, dataloader, layer_num, chunk_size=100):
        activations = []
        hook_name = utils.get_act_name('post', layer_num)
        
        def hook_fn(act, hook):
            if len(act.shape) == 3:
                act = act.reshape(-1, act.shape[-1])
            activations.append(act.detach().cpu())
        
        model.add_hook(hook_name, hook_fn)
        
        try:
            for i, batch in enumerate(dataloader):
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                _ = model(inputs)
                if (i+1) % chunk_size == 0:
                    yield torch.cat(activations, dim=0)
                    activations = []
        finally:
            model.reset_hooks()
        
        if activations:
            yield torch.cat(activations, dim=0)

    activation_generator = extract_mlp_activations_in_chunks(cfg.main_model, dataloader, layer_num, chunk_size)
    
    # Initialize the SAE with the correct input dimension
    first_batch = next(activation_generator)
    input_dim = first_batch.shape[-1]
    sae = AdaptiveSparseAutoencoder(encoding_dim).cuda()
    sae.initialize(input_dim)
    
    # Create a new generator that includes the first batch
    def new_generator():
        yield first_batch
        yield from extract_mlp_activations_in_chunks(cfg.main_model, dataloader, layer_num, chunk_size)
    
    train_sae(sae, new_generator())
    
    return sae