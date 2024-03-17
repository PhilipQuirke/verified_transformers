import re

from .useful_config import UsefulConfig


# Extends UsefulConfig with mathematics-specific info for "123456+123456=+0246912" style questions
class MathsConfig(UsefulConfig):

    def __init__(self):
        super().__init__()

        # Percent of questions that are multiplication, subtraction (rest are addition questions).
        self.perc_mult = 0 # e.g. 20
        self.perc_sub = 0 # e.g. 80

        # Save graphs to CoLab temp files as PDF or SVG. You can manually export temp files for re-use in papers.
        self.graph_file_suffix = "svg"
        
        self.n_digits = 6
        self.initialize_token_positions( self.question_tokens(), self.answer_tokens(), self.answer_meanings_ascend )


    def perc_add(self):
        return max(0, 100 - self.perc_mult - self.perc_sub)


    # The number of question tokens
    # This is also the token position of the first answer digit (which is a "+" or a  "-")
    def question_tokens(self):
        return self.n_digits*2 + 2

    def answer_tokens(self):
        return self.n_digits + 2

    def n_ctx(self):
        return self.question_tokens() + self.answer_tokens()


    # How many slices do we break the MLP layer up into?
    def mlp_slices(self):
        return 1 # Paper 2 used this granualarity
        # return self.n_heads * self.d_mlp_multiplier # Alternative for Paper 3?
    

    def parse_model_name(self):
        super().parse_model_name()
        
        match = re.search("d(\d)_", self.model_name)
        if match:
            self.n_digits = int(match.group(1))
            
        self.initialize_token_positions( self.question_tokens(), self.answer_tokens(), self.answer_meanings_ascend )  


    def file_config_prefix(self):
        op_prefix = 'mul' if self.perc_mult == 100 else 'sub' if self.perc_sub == 100 else 'add' if self.perc_add() == 100 else 'mix'

        return op_prefix + f'_d{self.n_digits}' + super().file_config_prefix()
