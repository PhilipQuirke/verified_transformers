import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import transformer_lens.utils as utils


def generate_encodings(model, sae, dataloader, layer_num, max_samples=10000):
    encodings = []
    sample_count = 0
    hook_name = utils.get_act_name('post', layer_num)

    def hook_fn(act, hook):
        nonlocal sample_count
        with torch.no_grad():
            encoded, _ = sae(act.cuda())
        encodings.append(encoded.cpu())
        sample_count += act.shape[0]

    model.add_hook(hook_name, hook_fn)

    try:
        with torch.no_grad():
            for batch in dataloader:
                _ = model(batch)
                if sample_count >= max_samples:
                    break
    finally:
        model.reset_hooks()

    encodings = torch.cat(encodings, dim=0)[:max_samples]
    print(f"Generated encodings shape: {encodings.shape}")
    return encodings

def visualize_encodings(encodings, perplexity=30, n_iter=250):
    num_positions = encodings.shape[1]
    num_features = encodings.shape[2]
    
    print(f"Analyzing {num_positions} positions, each with {num_features} features")

    for position in range(num_positions):
      
        # Extract encodings for the current position
        position_encodings = encodings[:, position, :]

        print(f"\nPosition {position}. Position encodings shape: {position_encodings.shape}")

        # Check for NaN or inf values
        if torch.isnan(position_encodings).any() or torch.isinf(position_encodings).any():
            print("Warning: NaN or inf values detected in encodings")
            position_encodings = torch.nan_to_num(position_encodings, nan=0.0, posinf=1e6, neginf=-1e6)

        # Convert to numpy and ensure float64 dtype for t-SNE
        encodings_np = position_encodings.numpy().astype(np.float64)

        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        try:
            encodings_2d = tsne.fit_transform(encodings_np)
        except ValueError as e:
            print(f"Error during t-SNE: {str(e)}")
            print(f"Encodings shape: {encodings_np.shape}")
            print(f"Encodings min: {np.min(encodings_np)}, max: {np.max(encodings_np)}")
            continue

        # Analyze sparsity. 80% is good.
        sparsity = (position_encodings == 0).float().mean()
        print(f"Sparsity of encoded representations at position {position}: {sparsity.item():.2%}")

        # Visualize t-SNE
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], c=encodings_np.mean(axis=1), cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title(f"t-SNE visualization of SAE encodings at position {position}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.show()

        # Visualize average activation
        avg_activation = position_encodings.abs().mean(dim=0)
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(avg_activation)), avg_activation)
        plt.title(f"Average Activation of Encoded Dimensions at position {position}")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Average Absolute Activation")
        plt.show()

def analyze_and_visualize_sae(cfg, sae, dataloader, layer_num=0, max_samples=10000, perplexity=30, n_iter=250):
    print("Generating encodings...")
    encodings = generate_encodings(cfg.main_model, sae, dataloader, layer_num, max_samples)
    
    print("Visualizing encodings...")
    visualize_encodings(encodings, perplexity, n_iter)


