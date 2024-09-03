import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import transformer_lens.utils as utils


# Extract MLP activations
def extract_mlp_activations(model, dataloader, layer_num):
    activations = []
    
    def hook_fn(act, hook):
        activations.append(act.detach())
    
    # Get the hook name
    hook_name = utils.get_act_name('post', layer_num)
    
    # Register the hook
    model.add_hook(hook_name, hook_fn)
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                _ = model(batch)
    finally:
        # model.remove_all_hooks()
        model.reset_hooks()
    
    
    return torch.cat(activations, dim=0)


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
def train_sae(sae, activations, batch_size=64, num_epochs=100, learning_rate=1e-3):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    dataloader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            encoded, decoded = sae(x)
            loss = sae.loss(x, encoded, decoded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")


def analyze_mlp_with_sae(cfg, dataloader, layer_num=0, encoding_dim=64):
    # Extract MLP activations
    mlp_activations = extract_mlp_activations(cfg.main_model, dataloader, layer_num)
    
    # Create and train SAE
    input_dim = mlp_activations.shape[1]
    sae = SparseAutoencoder(input_dim, encoding_dim)
    train_sae(sae, mlp_activations)
    
    return sae, mlp_activations

