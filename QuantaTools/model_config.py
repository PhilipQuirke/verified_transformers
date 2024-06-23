import re

from .useful_node import answer_name


# Base configuration class related to a Transformer model shape, training and analysis
class ModelConfig():

    def __init__(self):
        # Model name for models stored on HuggingFace
        self.model_name = ""
        
        # The transformer lens model
        self.main_model = None

        # Model shape configuration
        self.n_layers : int = 3 
        self.n_heads : int = 4 
        self.d_vocab : int = 15
        self.d_model : int = 510
        self.d_mlp_multiplier : int = 4
        self.d_head : int = 170
        self.act_fn = 'relu'

        # Training technique that adds 50% memory but speeds training by x10.
        # Refer https://github.com/ironjr/grokfast/tree/main
        self.grokfast = False
 
        # Batch size. Training often uses 64. Larger size used for speed during analysis e.g. 1M Qs
        self.batch_size : int = 512 
  
        # Optimizer
        self.n_training_steps : int = 15000
        self.weight_decay = 0.1
        self.lr = 0.00008

        # Before training was this model initialised with another existing model?
        self.insert_mode = 0 # 0=None 1=Init, 2=FreezeHeads 3=FreezeAll
        self.insert_late = False
        self.insert_n_layers = 2
        self.insert_n_heads = 3
        self.insert_training_seed = 372001
        self.insert_n_training_steps = 15000
        
        # Random seeds
        self.training_seed : int = 372001
        self.analysis_seed : int = 673023

        # Vocabulary: Map from each character to each token    
        self.char_to_token = {}

        # List of (short) strings representing the meaning of each token position.
        # For example D5, D4, D3, D2, D1, D0, +, D'5, D'4, D'3, D'2, D'1, D'0, =, A6, A5, A4, A3, A2, A1, A0
        # Used in node tag, in the column headings of quanta-maps, etc. 
        self.token_position_meanings = []
  
        self.initialize_token_positions( 12, 7, True ) # Random values (based on 5 digit addition)
    
        # Should we use the GPU (if any) to speed up processing?
        self.use_cuda = True

        # Format to save graphs to CoLab temp files. 
        # Temp files can then be manually exported for re-use in papers etc.
        self.graph_file_suffix = "pdf" # Can be pdf, svg, png or blank to suppress saving
    
 
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

       
    @property
    def n_ctx(self) -> int:
        return self.num_question_positions + self.num_answer_positions


    @property
    def d_mlp(self) -> int:
        return self.d_mlp_multiplier * self.d_model


    # Update n_digits, n_layers, n_heads, n_training_steps, training_seed, grokfast from model_name
    def parse_model_name(self):

        match = re.search(r"l(\d)_", self.model_name)
        if match:
            self.n_layers = int(match.group(1))


        match = re.search(r"h(\d)_", self.model_name)
        if match:
            self.n_heads = int(match.group(1))


        match = re.search(r"t(\d\d)K", self.model_name)
        if not match:
            match = re.search(r"t(\d)K", self.model_name)        
        if match:
            self.n_training_steps = int(match.group(1)) * 1000
    
            
        self.insert_mode = 0
        if "ins1_" in self.model_name :
            self.insert_mode = 1 # Initialised with some existing model before training
        elif "ins2_" in self.model_name :
            self.insert_mode = 2 # Initialised with existing model. Train & reset useful heads every 100 epochs
        elif "ins3_" in self.model_name :
            self.insert_mode = 3 # Initialised with existing model. Trained & reset useful heads & MLPs every 100 epochs
        elif "ins4_" in self.model_name :
            self.insert_mode = 4 # Initialised with "nodes with identified subtasks" from existing model. Train & reset useful heads every 100 epochs
     
        match = re.search(r"_s(\d\d\d\d\d\d)", self.model_name)
        if match:
            self.training_seed = int(match.group(1))

        self.grokfast = ("_gf" in self.model_name)


   # Update n_digits, n_layers, n_heads, n_training_steps, training_seed from model_name
    def parse_insert_model_name(self, insert_model_name):

        match = re.search(r"l(\d)_", insert_model_name)
        if match:
            self.insert_n_layers = int(match.group(1))

        match = re.search(r"h(\d)_", insert_model_name)
        if match:
            self.insert_n_heads = int(match.group(1))

        match = re.search(r"t(\d\d)K", insert_model_name)
        if not match:
            match = re.search(r"t(\d)K", insert_model_name)       
        if match:
            self.insert_n_training_steps = int(match.group(1)) * 1000
    
        match = re.search(r"_s(\d\d\d\d\d\d)", insert_model_name)
        if match:
            self.insert_training_seed = int(match.group(1))


    @property
    def short_config_description(self) -> str:       
        return f'_l{self.n_layers}_h{self.n_heads}'   
    

    @property
    def long_config_description(self) -> str:
        train_str = str(self.n_training_steps//1000) 
        return self.short_config_description + f'_t{train_str}K_s{self.training_seed}'


    @property
    def insert_config_description(self) -> str:
        return '' if self.insert_mode == 0 else f'ins{self.insert_mode}_' 
    

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_vocab": self.d_vocab,    
            "d_model": self.d_model,    
            "d_mlp_multiplier": self.d_mlp_multiplier,
            "d_head": self.d_head,
            "act_fn": self.act_fn,
            "grokfast": self.grokfast,
            "batch_size": self.batch_size,
            "n_training_steps": self.n_training_steps,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "insert_mode": self.insert_mode,
            "insert_late": self.insert_late,
            "insert_n_layers": self.insert_n_layers,
            "insert_n_heads": self.insert_n_heads,
            "insert_training_seed": self.insert_training_seed,
            "insert_n_training_steps": self.insert_n_training_steps,
            "training_seed": self.training_seed,
            "analysis_seed": self.analysis_seed,
        }


  