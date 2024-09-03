import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import transformer_lens.utils as utils
        

# Implement the SAE
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_weight=1e-5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        self.sparsity_weight = sparsity_weight
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def loss(self, x, encoded, decoded):
        mse_loss = nn.MSELoss()(decoded, x)
        sparsity_loss = torch.mean(torch.abs(encoded))
        return mse_loss + self.sparsity_weight * sparsity_loss
    

# Train the SAE 
def train_sae(sae, activation_generator, batch_size=64, num_epochs=100, learning_rate=1e-3):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for activations in activation_generator:
            dataloader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)
            
            for batch in dataloader:
                x = batch[0].cuda()  # Move to GPU
                optimizer.zero_grad()
                encoded, decoded = sae(x)
                loss = sae.loss(x, encoded, decoded)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/num_batches:.4f}")


def analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=32, chunk_size=1000):

    # Get a sample batch to determine input dimension
    sample_batch = next(iter(dataloader))
    
    def sample_hook(act, hook):
        return act  # Simply return the activation

    with torch.no_grad():
        sample_activation = cfg.main_model.run_with_hooks(
            sample_batch, 
            fwd_hooks=[(utils.get_act_name('post', layer_num), sample_hook)]
        )
    input_dim = sample_activation.shape[-1]
    
    # Create SAE
    sae = SparseAutoencoder(input_dim, encoding_dim).cuda()
    
    # Create a generator for activations
    def extract_mlp_activations_in_chunks(model, dataloader, layer_num, chunk_size=1000):
        activations = []
        hook_name = utils.get_act_name('post', layer_num)
        
        def hook_fn(act, hook):
            activations.append(act.detach().cpu())  # Move to CPU immediately
        
        model.add_hook(hook_name, hook_fn)
        
        try:
            with torch.no_grad():
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
