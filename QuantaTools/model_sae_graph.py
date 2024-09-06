import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import transformer_lens.utils as utils

from QuantaTools.model_sae import AdaptiveSparseAutoencoder 


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


def visualize_encodings_by_position(encodings, perplexity=30, n_iter=250):
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
        plt.figure(figsize=(4, 3))
        scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], c=encodings_np.mean(axis=1), cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.title(f"t-SNE visualization of SAE encodings at position {position}")
        plt.xlabel("t-SNE feature 1")
        plt.ylabel("t-SNE feature 2")
        plt.show()

        # Visualize average activation
        avg_activation = position_encodings.abs().mean(dim=0)
        plt.figure(figsize=(12, 3))
        plt.bar(range(len(avg_activation)), avg_activation)
        plt.title(f"Average Activation of Encoded Dimensions at position {position}")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Average Absolute Activation")
        plt.show()


def visualize_sae_features_by_position(sae, top_n=5, figsize=(20, 4)):
    decoder_weights = sae.decoder.weight.data.cpu().numpy()
    input_dim, encoding_dim = decoder_weights.shape

    # Try to infer the number of positions
    possible_positions = [pos for pos in range(1, input_dim + 1) if input_dim % pos == 0]
    
    if not possible_positions:
        raise ValueError(f"Cannot determine number of positions. Input dimension {input_dim} has no suitable divisors.")

    print(f"Possible number of positions: {possible_positions}")
    num_positions = max(possible_positions)
    features_per_position = input_dim // num_positions

    print(f"Assuming {num_positions} positions with {features_per_position} features per position")

    try:
        reshaped_weights = decoder_weights.reshape(num_positions, features_per_position, encoding_dim)
    except ValueError as e:
        print(f"Error reshaping weights: {e}")
        print(f"Decoder weights shape: {decoder_weights.shape}")
        print(f"Attempted reshape: ({num_positions}, {features_per_position}, {encoding_dim})")
        return

    # Calculate the L2 norm of each feature for each position
    feature_norms = np.linalg.norm(reshaped_weights, axis=1)

    # Get the indices of the top N features for each position
    top_features = np.argsort(-feature_norms, axis=1)[:, :top_n]

    # Plotting
    fig, axes = plt.subplots(1, num_positions, figsize=figsize, sharey=True)
    if num_positions == 1:
        axes = [axes]

    for pos in range(num_positions):
        ax = axes[pos]
        for i, feature in enumerate(top_features[pos]):
            ax.bar(i, feature_norms[pos, feature], label=f'Feature {feature}')
            ax.text(i, feature_norms[pos, feature], f'{feature}', ha='center', va='bottom')
        
        ax.set_title(f'Position {pos+1}')
        ax.set_xticks([])
        if pos == 0:
            ax.set_ylabel('L2 Norm')

    plt.tight_layout()
    plt.show()

    # Additional visualization: heatmap of all features
    plt.figure(figsize=(20, 10))
    plt.imshow(feature_norms.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='L2 Norm')
    plt.title('Heatmap of Feature Importance Across Positions')
    plt.xlabel('Position')
    plt.ylabel('Feature')
    plt.show()


def analyze_and_visualize_sae(cfg, sae, dataloader, layer_num=0, max_samples=10000, perplexity=30, n_iter=250):
    print("Generating encodings...")
    encodings = generate_encodings(cfg.main_model, sae, dataloader, layer_num, max_samples)
    
    print("Visualizing encodings...")
    visualize_encodings_by_position(encodings, perplexity, n_iter)
    
    print("Visualizing SAE features by position...")
    visualize_sae_features_by_position(sae)

