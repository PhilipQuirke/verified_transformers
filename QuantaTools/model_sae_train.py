import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import transformer_lens.utils as utils
import hashlib
import itertools
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

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
            
    assert cfg.main_model is not None
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def power_of_two(n):
    return 2 ** int(n)

def generate_compact_filename(params):
    # Create a compact string representation of parameters
    param_str = "_".join(f"{k[:3]}{v:.3g}" if isinstance(v, float) else f"{k[:3]}{v}" 
                         for k, v in sorted(params.items()))
    
    # Hash the parameter string to ensure filename length is manageable
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    return f"sae_{param_hash}"

def save_json(file_name, data_to_save, save_folder):
    json_path = os.path.join(save_folder, f"{file_name}.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4, cls=NumpyEncoder)
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")
    except json.JSONDecodeError as e:
        print(f"An error occurred while encoding JSON: {e}")

def load_json(file_name, save_folder):
    json_path = os.path.join(save_folder, f"{file_name}.json")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"An error occurred while decoding JSON: {e}")
        return None

def optimize_sae_hyperparameters(cfg, dataloader, layer_num=0, param_space=None, save_folder=None, n_calls=50):
    if param_space is None:
        param_space = [
            Integer(5, 9, name='encoding_dim_exp'),  # 2^5=32 to 2^9=512
            Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
            Real(0.001, 0.1, prior='log-uniform', name='sparsity_target'),
            Real(1e-3, 1.0, prior='log-uniform', name='sparsity_weight'),
            Real(1e-6, 1e-3, prior='log-uniform', name='l1_weight'),
            Integer(5, 20, name='num_epochs'),
            Integer(2, 3, name='patience')
        ]

    @use_named_args(param_space)
    def objective(**params):
        # Convert encoding_dim_exp to actual encoding_dim
        actual_params = params.copy()
        actual_params['encoding_dim'] = power_of_two(actual_params.pop('encoding_dim_exp'))
        
        file_name = generate_compact_filename(actual_params)

        # Check if this experiment has already been run
        this_json = load_json(file_name, save_folder)
        if this_json is not None:
            print(f"\nLoading experiment: {file_name}. Score: {this_json['score']:.4f}, Neurons used: {this_json['neurons_used']}, Params {actual_params}")
            return this_json['score']

        print(f"\nRunning Experiment: {file_name} Params: {actual_params}")
        this_sae, score, avg_loss, avg_sparsity, neurons_used = analyze_mlp_with_sae(
            cfg, 
            dataloader, 
            layer_num=layer_num, 
            **actual_params
        )

        this_json = {
            "parameters": actual_params,
            "score": score,
            "avg_loss": avg_loss, 
            "avg_sparsity": avg_sparsity,
            "neurons_used": neurons_used
        }

        if save_folder is not None:
            model_path = os.path.join(save_folder, f"{file_name}.pth")
            torch.save(this_sae.state_dict(), model_path)
            save_json(file_name, this_json, save_folder)

        print(f"Score: {score:.4f}, "
              f"Loss: {avg_loss:.4f} "
              f"Sparsity: {avg_sparsity:.4f}, "
              f"Neurons: {neurons_used}, "
              f"Params: {actual_params}")

        return score

    result = gp_minimize(objective, param_space, n_calls=n_calls, random_state=42)

    # Convert the best encoding_dim_exp back to actual encoding_dim
    best_params = dict(zip([dim.name for dim in param_space], result.x))
    best_params['encoding_dim'] = power_of_two(best_params.pop('encoding_dim_exp'))
    best_score = result.fun

    return best_score, generate_compact_filename(best_params), best_params