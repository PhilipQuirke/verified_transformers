import re

from QuantaTools.useful_node import position_name
from QuantaTools.algo_config import AlgoConfig


# Extends UsefulConfig with mathematics-specific info for "123456+123456=+0246912" style questions
class MathsConfig(AlgoConfig):

    # Constructor
    def __init__(self):
        super().__init__()

        # Percent of questions that are multiplication, subtraction (rest are addition questions) in training batches
        self.perc_mult : int = 0 # e.g. 20
        self.perc_sub : int = 0 # e.g. 80 

        # Number of digits in the question e.g. 123456+123456=+0246912
        self.n_digits : int = 6
        self.initialize_maths_token_positions()     

        # Dictionary of test maths questions based on the ST8, ST9, ST10 categorisation. Indexed by (digit, operator).
        self.tricase_questions_dict = {}

        # More granular tricase_questions, indexed by (digit, operator, qtype).
        # Makes easier how we mix and match tricase data from different qtypes.
        self.customized_tricase_questions_dict = {}
      

    @property
    # percentage of addition questions
    def perc_add(self) -> int:
        return max(0, 100 - self.perc_mult - self.perc_sub)
    

    # Based on n_digits, set the number of question and answer tokens in the context 
    def initialize_maths_token_positions(self):
        self.initialize_token_positions( 
            self.n_digits*2 + 2,  # Plus 2 for operator (+ or -) and equals (=) sign
            self.n_digits + 2, # Plus 2 for answer sign (+ or -) and answer digits (adding two 5 digits numbers gives a 6 digit answer )
            False ) 
        
        if self.perc_add == 100:
            # The first answer token is always "+"
            self.token_position_meanings[self.num_question_positions] = "+"


    @property
    # How many slices do we break the MLP layer up into?
    def mlp_slices(self) -> int:
        return 1 # Paper 2 used this granularity
        # return self.n_heads * self.d_mlp_multiplier # Alternative for Paper 3?
  

    # Maths question and answer token position meanings are D5, .., D0, *, D5', .., D0', =, A7, A6, .., A0      
    # Convert D0 to P5, D1 to P4, D2 to P3 in 6 digit addition
    def dn_to_position_name(self, n):
        return position_name(self.n_digits - 1 - n) 
    # Convert D'0 to P10, D'1 to P9, D'2 to P8, etc in 6 digit addition
    def ddn_to_position_name(self, n):
        return position_name(2 * self.n_digits - n) 
    # Convert A0 to P20, A1 to P19, A2 to P18, etc in 6 digit addition
    def an_to_position_name(self, n):
        return position_name(self.n_ctx - 1 - n)
    # Position of the operator (+, -, * or /)
    def op_position_name(self):
        return position_name(self.n_digits)


    # Parse the model name to extract the number of digits in question
    def parse_model_name(self):
        super().parse_model_name()
        
        
        if "sub_" in self.model_name :
            # Subtraction model
            self.perc_sub = 100
            self.perc_mult = 0
            
        elif "add_" in self.model_name :
            # Addition model
            self.perc_sub = 0
            self.perc_mult = 0
            
        elif "mul_" in self.model_name :
            # Multiplication model
            self.perc_sub = 0
            self.perc_mult = 100
            
        elif "mix_" in self.model_name :
            # Mixed (addition and subtraction) model. 
            # Train on 66% sub and 33% add question batches, as sub is harder to learn than add
            self.perc_sub = 66
            self.perc_mult = 0
            if self.model_name.startswith("ins") :
                  # Mixed model initialised with an addition model (using insert mode 1, 2 or 3)
                  self.perc_sub = 80 # Train on 80% subtraction and 20% addition question batches
                  
        elif "mas_" in self.model_name :
            # Multiplication, addition and subtraction model. 
            # Train on 50% mult, 30% sub and 20% add question batches, as mult is harder to learn than sub which is harder than add.
            # Use this split even if we are initialising with a mixed (addition and subtraction) model.
            self.perc_sub = 30
            self.perc_mult = 50


        match = re.search(r"d(\d)_", self.model_name)
        if not match:
            match = re.search(r"d(\d\d)_", self.model_name)
        if match:
            self.n_digits = int(match.group(1))
            

        # n_digits may have changed 
        self.initialize_maths_token_positions()  


    # Parse the insert model name to extract the number of insert digits 
    def parse_insert_model_name(self):
        super().parse_insert_model_name()
                    
        match = re.search(r"d(\d)_", self.insert_model_name)
        if not match:
            match = re.search(r"d(\d\d)_", self.insert_model_name)
        if match:
            self.insert_n_digits = int(match.group(1))
            

    @property
    # Extend "l2_h3_t15K" with number of digits in question to give "_d5_l2_h3_t15K
    def short_config_description(self) -> str:       
        return f'_d{self.n_digits}' + super().short_config_description      
    

    @property
    # Return string stating whether we are doing multiplication, subtraction, addition or a mix
    def op_config_description(self) -> str:
        return 'mul' if self.perc_mult == 100 else 'sub' if self.perc_sub == 100 else 'add' if self.perc_add == 100 else 'mix'    
    

    @property
    # Return string like "ins1_mix_d6_l3_h4_t40K"
    def file_config_prefix(self) -> str:
        return self.insert_config_description + self.op_config_description + self.long_config_description
    

    # Return integer 444444 for 6 digit number
    @staticmethod
    def repeat_digit_n(digit, n):
        if n <= 0:
            return 0
        
        return int(str(digit) * n)
    
    # Return integer 444444 for 6 digit number
    def repeat_digit(self, digit):
        return MathsConfig.repeat_digit_n(digit, self.n_digits)


    # Return a dictionary of all the model configuration parameters
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'perc_mult': self.perc_mult,
            'perc_sub': self.perc_sub,
            'perc_add': self.perc_add, # Readonly
            'n_digits': self.n_digits,
            'mlp_slices': self.mlp_slices, # Readonly
            'op_position_name': self.op_position_name() # Readonly
        })
        return base_dict


    # Set attributes from JSON data, using default values if attributes are missing
    def init_from_json(self, data):
        super().init_from_json(data)
        self.perc_mult = data.get('perc_mult', self.perc_mult)
        self.perc_sub = data.get('perc_sub', self.perc_sub)
        self.n_digits = data.get('n_digits', self.n_digits)
        

    def sanity_check(self):
        super().sanity_check()        
        assert(self.perc_mult >= 0)
        assert(self.perc_sub >= 0)
        assert(self.n_digits > 0)

