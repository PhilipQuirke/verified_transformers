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
    optimizer = None
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for activations in activation_generator:
            dataloader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)
            
            for batch in dataloader:
                x = batch[0].cuda()  # Move to GPU
                x.requires_grad_(True)  # Enable gradient computation
                
                if optimizer is None:
                    sae(x)  # This will initialize the encoder and decoder if not already done
                    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
                
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
    # Create SAE
    sae = AdaptiveSparseAutoencoder(encoding_dim).cuda()
    
    # Create a generator for activations
    def extract_mlp_activations_in_chunks(model, dataloader, layer_num, chunk_size=100):
        activations = []
        hook_name = utils.get_act_name('post', layer_num)
        
        def hook_fn(act, hook):
            # Flatten the activation if it's 3-dimensional
            if len(act.shape) == 3:
                act = act.reshape(-1, act.shape[-1])
            activations.append(act.clone())  # Use clone() instead of detach()
        
        model.add_hook(hook_name, hook_fn)
        
        try:
            for i, batch in enumerate(dataloader):
                _ = model(batch)
                if (i+1) % chunk_size == 0:
                    yield torch.cat(activations, dim=0)
                    activations = []  # Clear the list to free memory
        finally:
            model.reset_hooks()
        
        if activations:  # Yield any remaining activations
            yield torch.cat(activations, dim=0)

    activation_generator = extract_mlp_activations_in_chunks(cfg.main_model, dataloader, layer_num, chunk_size)
    
    # Train SAE
    train_sae(sae, activation_generator)
    
    return sae
