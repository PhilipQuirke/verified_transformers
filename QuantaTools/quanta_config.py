# Main configuration class for main model creation and training
class QuantaConfig():

  def __init__(self):
    # Model name for models stored on HuggingFace
    self.model_name = ""
  
    # Model shape configuration
    self.n_layers = 3 
    self.n_heads = 4 
    self.d_vocab = 15
    self.d_model = 510
    self.d_mlp_multiplier = 4
    self.d_head = 170
    self.act_fn = 'relu'
 
    # Batch size. Training often 64. Larger size used for speed during analysis
    self.batch_size = 512 
  
    # Optimizer
    self.n_training_steps = 40000
    self.weight_decay = 0.00008
    self.lr = 0.1
  
    # Random seeds
    self.training_seed = 372001
    self.analysis_seed = 673023
   
    def d_mlp(self):
      return d_mlp_multiplier * d_model
