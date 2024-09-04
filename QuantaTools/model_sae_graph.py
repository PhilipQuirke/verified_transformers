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

    return torch.cat(encodings, dim=0)[:max_samples]


def visualize_encodings(encodings, perplexity=30, n_iter=1000):
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    encodings_2d = tsne.fit_transform(encodings.numpy())

    # Visualize
    plt.figure(figsize=(12, 10))
    plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], alpha=0.5)
    plt.title(f"t-SNE visualization of SAE encodings (perplexity={perplexity})")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.colorbar()
    plt.show()

    # Analyze sparsity
    sparsity = (encodings == 0).float().mean()
    print(f"Sparsity of encoded representations: {sparsity.item():.2%}")

    # Visualize average activation
    avg_activation = encodings.abs().mean(dim=0)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(avg_activation)), avg_activation)
    plt.title("Average Activation of Encoded Dimensions")
    plt.xlabel("Encoding Dimension")
    plt.ylabel("Average Absolute Activation")
    plt.show()


def analyze_and_visualize_sae(cfg, sae, dataloader, layer_num=0, max_samples=10000, perplexity=30, n_iter=1000):
    print("Generating encodings...")
    encodings = generate_encodings(cfg.main_model, sae, dataloader, layer_num, max_samples)
    
    print("Visualizing encodings...")
    visualize_encodings(encodings, perplexity, n_iter)

