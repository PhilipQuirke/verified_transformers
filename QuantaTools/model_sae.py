import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import transformer_lens.utils as utils
import itertools


class AdaptiveSparseAutoencoder(nn.Module):
    def __init__(self, encoding_dim, input_dim, sparsity_weight=1e-5):
        super().__init__()

        # Number of neurons in the hidden layer of the autoencoder. Represents the compressed representation of the input data.
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim  
        self.sparsity_weight = sparsity_weight
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
    

def train_sae(sae, activation_generator, num_epochs=100, learning_rate=1e-3):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for activations in activation_generator:
            x = activations.cuda()
            x.requires_grad_(True)
            
            optimizer.zero_grad()
            encoded, decoded = sae(x)
            loss = sae.loss(x, encoded, decoded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/num_batches:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")


def analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=32):
    def generate_mlp_activations(main_model, dataloader, layer_num):
        hook_name = utils.get_act_name('post', layer_num)
        activations = [] 

        def store_activations_hook(act, hook):
            if len(act.shape) == 3:
                act = act.reshape(-1, act.shape[-1])
            activations.append(act.detach().cpu())
    
        try:
            main_model.add_hook(hook_name, store_activations_hook)
            for batch in dataloader:          
                _ = main_model(batch)
                yield torch.cat(activations, dim=0)
                activations = []                                
        finally:
            main_model.reset_hooks()

    activation_generator = generate_mlp_activations(cfg.main_model, dataloader, layer_num)
    
    sample_batch = next(activation_generator)
    input_dim = sample_batch.shape[-1]
    print("Input Dim", input_dim, "Encoding Dim", encoding_dim)
    print("Activation batch shape", sample_batch.shape, "Sample:", sample_batch[0])

    sae = AdaptiveSparseAutoencoder(encoding_dim, input_dim).cuda()

    train_sae(sae, activation_generator)
    
    return sae