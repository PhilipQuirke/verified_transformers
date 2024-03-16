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

    # Vocabulary: Map from each character to each token    
    self.char_to_token = {}

    # List of (short) strings representing the meaning of each token position.
    # For example D5, D4, D3, D2, D1, D0, +, D'5, D'4, D'3, D'2, D'1, D'0, =, A6, A5, A4, A3, A2, A1, A0
    # Used in node tag, in the column headings of quanta-maps, etc. 
    self.token_position_meanings = []
  
    self.initialize_token_positions(12, 6,True)
    
 
  def initialize_token_positions(self, num_question_positions, num_answer_positions, answer_meanings_ascend ):
    # The number of "question" (input) token positions e.g. len("12340+12340=")    
    self.num_question_positions = num_question_positions

    # The number of "answer" (output) token positions  e.g. len("+024680") 
    self.num_answer_positions = num_answer_positions

    # Do we name the answer tokens as A5, A4, A3, A2, A1, A0 or A0, A1, A2, A3, A4, A5?
    self.answer_meanings_ascend = answer_meanings_ascend   
    
    self.default_token_position_meanings()


  # Default list of strings representing the token positions meanings
  def default_token_position_meanings(self):
    self.token_position_meanings = []
    for i in range(self.num_question_positions):
      self.token_position_meanings += ["P"+str(i)]
    for i in range(self.num_answer_positions):
      self.token_position_meanings += [answer_name(i if self.answer_meanings_ascend else self.num_answer_positions - i - 1 )]


  def d_mlp(self):
    return d_mlp_multiplier * d_model
