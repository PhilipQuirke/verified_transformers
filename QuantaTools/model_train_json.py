import json
from typing import List


# The training colab creates a json file containing this information
class TrainingJsonConfig:
    def __init__(self, n_layers, n_heads, d_vocab, d_mlp, d_head, training_seed, n_digits, n_ctx, act_fn, batch_size, n_training_steps, lr, weight_decay, perc_mult, perc_sub, insert_late, insert_mode, insert_n_layers, insert_n_heads, insert_training_seed, insert_n_training_steps):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_vocab = d_vocab
        self.d_mlp = d_mlp
        self.d_head = d_head
        self.training_seed = training_seed
        self.n_digits = n_digits
        self.n_ctx = n_ctx
        self.act_fn = act_fn
        self.batch_size = batch_size
        self.n_training_steps = n_training_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.perc_mult = perc_mult
        self.perc_sub = perc_sub
        self.insert_late = insert_late
        self.insert_mode = insert_mode
        self.insert_n_layers = insert_n_layers
        self.insert_n_heads = insert_n_heads
        self.insert_training_seed = insert_training_seed
        self.insert_n_training_steps = insert_n_training_steps


# The training colab creates a json file containing this information
class TrainingJsonData:
    def __init__(self, config: TrainingJsonConfig, avg_final_loss: float, final_loss: float, training_loss: List[float]):
        self.config = config
        self.avg_final_loss = avg_final_loss
        self.final_loss = final_loss
        self.training_loss = training_loss


def load_training_json(file_path: str) -> TrainingJsonData:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    config_data = data['Config']
    config = TrainingJsonConfig(**config_data)
    
    avg_final_loss = data['AvgFinalLoss']
    final_loss = data['FinalLoss']
    training_loss = data['TrainingLoss']
    
    training_data = TrainingJsonData(config, avg_final_loss, final_loss, training_loss)
    return training_data

