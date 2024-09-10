import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import transformer_lens.utils as utils
import itertools
from sklearn.model_selection import ParameterGrid
import numpy as np

from QuantaTools.model_sae import AdaptiveSparseAutoencoder, save_sae_to_huggingface


def train_sae_epoch(sae, activation_generator, epoch, learning_rate, max_grad_norm=1.0):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
  
    total_loss = 0
    total_mse = 0
    total_sparsity_penalty = 0
    total_l1_penalty = 0
    total_sparsity = 0    
    num_batches = 0
    
    neuron_activity = torch.zeros(sae.encoding_dim).cuda()
    
    for activations in activation_generator:
        x = activations.cuda()
        x.requires_grad_(True)
        
        optimizer.zero_grad()
        encoded, decoded = sae(x)

        loss, mse, sparsity_penalty, l1_penalty = sae.loss(x, encoded, decoded)
        
        if torch.isfinite(loss):
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
            
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse
            total_sparsity_penalty += sparsity_penalty
            total_l1_penalty += l1_penalty
            batch_sparsity = (encoded != 0).float().mean().item()
            total_sparsity += batch_sparsity
            
            # Summarise which neurons have been active on at least one question in the batch
            neuron_activity = torch.logical_or(neuron_activity, torch.any(encoded != 0, dim=0))
            
            num_batches += 1
        else:
            print(f"Skipping batch due to non-finite loss: {loss.item()}")
    
   
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_sparsity_penalty = total_sparsity_penalty / num_batches
        avg_l1_penalty = total_l1_penalty / num_batches
        avg_sparsity = total_sparsity / num_batches       
        neurons_used = torch.sum(neuron_activity).item()

        return avg_loss, avg_mse, avg_sparsity_penalty, avg_l1_penalty, avg_sparsity, neurons_used
    
    return total_loss, total_mse, 1, 1, 1, 100


def analyze_mlp_with_sae(
        cfg, 
        dataloader, 
        layer_num=0, 
        encoding_dim=512, 
        learning_rate=1e-3, 
        sparsity_target=0.05, 
        sparsity_weight=1e-3, 
        l1_weight=1e-4,        
        early_stopping_threshold=1e-4, 
        num_epochs=10, 
        patience=2, 
        save_directory=None):
    
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

    # Calculate a score that balances loss, sparsity and interpretability
    def get_score(sae, avg_loss, avg_sparsity, neurons_used):        
        if torch.isfinite(torch.tensor(avg_loss)):
            fraction_neurons_active = 1.0 * neurons_used / sae.encoding_dim
            return ( avg_loss * 100 + # Penalty for high loss. Loss is can be ~0.02 hence the scaling factor.
                    avg_sparsity + # Sparsity is based on the number of active neurons. Penalize a high value. In range [0, 1]
                    fraction_neurons_active ) # Penalize a high number of active neurons. In range [0, 1]
        return float('inf')   

    def print_results(sae, epoch, avg_loss, avg_mse, avg_sparsity_penalty, avg_l1_penalty, avg_sparsity, neurons_used):
        print(f"Epoch: {epoch+1}, Score: {get_score(sae, avg_loss, avg_sparsity, neurons_used):.4f}, "
            f"Loss: {avg_loss:.4f}, MSE: {avg_mse:.4f}, "
            f"Sparsity Penalty: {avg_sparsity_penalty:.4f}, "
            f"L1 Penalty: {avg_l1_penalty:.4f}, "
            f"Sparsity: {avg_sparsity:.2%}, "  # Measure of activation frequency
            f"Neurons used: {neurons_used}/{sae.encoding_dim} ({neurons_used/sae.encoding_dim:.2%})") # Fraction of neurons used on at least one question
            

    activation_generator = generate_mlp_activations(cfg.main_model, dataloader, layer_num)
    sample_batch = next(activation_generator)
    input_dim = sample_batch.shape[-1]

    sae = AdaptiveSparseAutoencoder(encoding_dim, input_dim, sparsity_target, sparsity_weight, l1_weight).cuda()

    prev_avg_loss = float('inf')
    prev_avg_sparsity = 0
    best_avg_loss = float('inf')
    best_avg_sparsity = 0
    neurons_used = 0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        activation_generator = generate_mlp_activations(cfg.main_model, dataloader, layer_num)
        avg_loss, avg_mse, avg_sparsity_penalty, avg_l1_penalty, avg_sparsity, neurons_used = train_sae_epoch(sae, activation_generator, epoch, learning_rate)
        
        if torch.isfinite(torch.tensor(avg_loss)):
            loss_change = abs(avg_loss - prev_avg_loss)
            sparsity_change = abs(avg_sparsity - prev_avg_sparsity)
            
            if loss_change < early_stopping_threshold and sparsity_change < early_stopping_threshold:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no significant improvement.")
                    print_results(sae, epoch, avg_loss, avg_mse, avg_sparsity_penalty, avg_l1_penalty, avg_sparsity, neurons_used)
                    break
            else:
                no_improvement_count = 0
            
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_avg_sparsity = avg_sparsity
            
            prev_avg_loss = avg_loss
            prev_avg_sparsity = avg_sparsity
        else:
            print(f"Skipping epoch {epoch+1} due to non-finite loss.")
   
        if epoch % 5 == 0 or epoch == num_epochs-1:
            print_results(sae, epoch, avg_loss, avg_mse, avg_sparsity_penalty, avg_l1_penalty, avg_sparsity, neurons_used)
    
        # Calculate a score that balances loss, sparsity and interpretability
        score = get_score(sae, avg_loss, avg_sparsity, neurons_used)     
            
    if save_directory is not None:
        save_sae_to_huggingface(sae, save_directory)
        
    return sae, score, best_avg_loss, best_avg_sparsity, neurons_used


# Want an SAE with:
# - Low Avg Loss and Avg MSE (good reconstruction)
# - Lower sparsity (fewer neurons firing per prediction)
# - Lower number of Final Active Neurons (easier interpretability)
def optimize_sae_hyperparameters(cfg, dataloader, layer_num=0, param_grid=None, save_folder=None):
    
    # Define the hyperparameter grid
    if param_grid is None:
        param_grid = {
            'encoding_dim': [32, 64, 128, 256, 512],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'sparsity_target': [0.001, 0.005, 0.01, 0.05, 0.1],
            'sparsity_weight': [1e-3, 1e-2, 1e-1, 1.0],
            'l1_weight': [1e-6, 1e-5, 1e-4, 1e-3],       
            'num_epochs': [20],
            'patience': [2]
        }



    # Save the params and results
    def save_json(file_name, data_to_save):
        json_path = os.path.join(save_folder, file_name)
        try:
            with open(json_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
        except IOError as e:
            print(f"An error occurred while writing to the file: {e}")
        except json.JSONDecodeError as e:
            print(f"An error occurred while encoding JSON: {e}")



    # Generate all combinations of hyperparameters
    grid = ParameterGrid(param_grid)

    best_sae = None
    best_json = None

    experiment_num = 0
    for params in grid:
        print()
        print(f"Testing parameters: {params}")
        this_sae, score, avg_loss, avg_sparsity, neurons_used = analyze_mlp_with_sae(
            cfg, 
            dataloader, 
            layer_num=layer_num, 
            encoding_dim=params['encoding_dim'],
            learning_rate=params['learning_rate'],
            sparsity_target=params['sparsity_target'],
            sparsity_weight=params['sparsity_weight'],
            l1_weight=params['l1_weight'],
            num_epochs=params['num_epochs'],
            patience=params['patience'],
        )

        this_json = {
            "parameters": params,
            "score": score,
            "neurons_used": neurons_used
        }

        if save_folder is not None:
            # Save the trained SAE, params and results
            model_path = os.path.join(save_folder, f"sae{experiment_num}_model.pth")
            torch.save(this_sae.state_dict(), model_path)
            save_json( f"sae{experiment_num}_params.json", this_json)

        if best_json is None or score < best_json["score"]:
            best_json = this_json
            best_sae = this_sae
            print(f"Better: Score: {score:.4f}, Neurons: {neurons_used}, Params {params}")

        experiment_num += 1

    if save_folder is not None:
        # Save the best trained SAE, params and results
        model_path = os.path.join(save_folder, f"sae_best_model.pth")
        torch.save(best_sae.state_dict(), model_path)
        save_json( "sae_best_params.json", best_json)

    return best_sae, best_json["score"], best_json["neurons_used"], best_json["parameters"]

