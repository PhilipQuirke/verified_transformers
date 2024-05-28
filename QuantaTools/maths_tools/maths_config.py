import re

from QuantaTools.useful_node import position_name
from QuantaTools.algo_config import AlgoConfig


# Extends UsefulConfig with mathematics-specific info for "123456+123456=+0246912" style questions
class MathsConfig(AlgoConfig):

    # Constructor
    def __init__(self):
        super().__init__()

        # Percent of questions that are multiplication, subtraction (rest are addition questions).
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
      

    # percentage of addition questions
    def perc_add(self):
        return max(0, 100 - self.perc_mult - self.perc_sub)
    

    # Based on n_digits, set the number of question and answer tokens in the context 
    def initialize_maths_token_positions(self):
        self.initialize_token_positions( 
            self.n_digits*2 + 2,  # Plus 2 for operator (+ or -) and equals (=) sign
            self.n_digits + 2, # Plus 2 for answer sign (+ or -) and answer digits (adding two 5 digits numbers gives a 6 digit answer )
            False ) 
        
        if self.perc_add() == 100:
            # The first answer token is always "+"
            self.token_position_meanings[self.num_question_positions] = "+"


    # How many slices do we break the MLP layer up into?
    def mlp_slices(self):
        return 1 # Paper 2 used this granualarity
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
        return position_name(self.n_ctx() - 1 - n)
    # Position of the operator (+, -, * or /)
    def op_position_name(self):
        return position_name(self.n_digits)


    # Parse the model name to extract the number of digits in question
    def parse_model_name(self):
        super().parse_model_name()
        
        match = re.search(r"d(\d)_", self.model_name)
        if match:
            self.n_digits = int(match.group(1))
        else:
            match = re.search(r"d(\d\d)_", self.model_name)
            if match:
                self.n_digits = int(match.group(1))
            
        # n_digits may have changed 
        self.initialize_maths_token_positions()  


    # Parse the model name to extract the number of digits in question
    def parse_insert_model_name(self, insert_model_name):
        super().parse_insert_model_name(insert_model_name)
        
        match = re.search(r"d(\d)_", insert_model_name)
        if match:
            self.insert_n_digits = int(match.group(1))
        else:
            match = re.search(r"d(\d\d)_", insert_model_name)
            if match:
                self.insert_n_digits = int(match.group(1))
                

    # Extend "l2_h3_t15K" with number of digits in question to give "_d5_l2_h3_t15K
    def short_config_description(self):       
        return f'_d{self.n_digits}' + super().short_config_description()      
    

    # Return string stating whether we are doing multiplication, subtraction, addition or a mix
    def op_config_description(self):
        return 'mul' if self.perc_mult == 100 else 'sub' if self.perc_sub == 100 else 'add' if self.perc_add() == 100 else 'mix'    
    

    # Return string like "ins1_mix_d6_l3_h4_t40K"
    def file_config_prefix(self):
        return self.insert_config_description() + self.op_config_description() + self.long_config_description()
    

    # Return integer 444444 for 6 digit number
    @staticmethod
    def repeat_digit_n(digit, n):
        if n <= 0:
            return 0
        
        return int(str(digit) * n)
    
    # Return integer 444444 for 6 digit number
    def repeat_digit(self, digit):
        return MathsConfig.repeat_digit_n(digit, self.n_digits)


