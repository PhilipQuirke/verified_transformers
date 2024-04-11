import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.useful_node import UsefulNode

from QuantaTools.quanta_constants import QType
from QuantaTools.quanta_map_impact import sort_unique_digits

from QuantaTools.maths_config import MathsConfig
from QuantaTools.maths_constants import MathsToken, MathsBehavior
from QuantaTools.maths_utilities import set_maths_vocabulary, int_to_answer_str, tokens_to_unsigned_int
from QuantaTools.maths_test_questions import make_maths_test_questions_and_answers, make_maths_questions_and_answers
from QuantaTools.maths_test_questions import make_maths_s0_questions_and_answers, make_maths_s1_questions_and_answers, make_maths_s2_questions_and_answers, make_maths_s3_questions_and_answers, make_maths_s4_questions_and_answers, make_maths_s5_questions_and_answers
from QuantaTools.maths_test_questions import make_maths_m0_questions_and_answers, make_maths_m1_questions_and_answers, make_maths_m2_questions_and_answers, make_maths_m3_questions_and_answers
from QuantaTools.maths_test_questions import make_maths_n1_questions_and_answers, make_maths_n2_questions_and_answers, make_maths_n3_questions_and_answers, make_maths_n4_questions_and_answers


class TestUseful(unittest.TestCase):


    def test_useful_node(self):
        
        useful_node = UsefulNode(1, 2, True, 3, [])

        # Add 6 distinct tags
        useful_node.add_tag( "Major0", "Minor1" )
        useful_node.add_tag( "Major0", "Minor2" )
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major1", "Minor2" )
        useful_node.add_tag( "Major2", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)
        
        # Add 2 duplicate tags
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)

        # Remove 2 tags
        useful_node.reset_tags( "Major2" )
        self.assertEqual( len(useful_node.tags), 4)

        # Remove 1 tag
        useful_node.reset_tags( "Major0", "Minor2" )
        self.assertEqual( len(useful_node.tags), 3)

        # Remove all tags
        useful_node.reset_tags( "" )
        self.assertEqual( len(useful_node.tags), 0)


class TestMaths(unittest.TestCase):


    def test_int_to_answer_str(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        self.assertEqual( int_to_answer_str(cfg, 1234), "+0001234" )

        
    def test_tokens_to_unsigned_int(self):
        q = [0,1,2,3,4,5]
        offset = 0
        digits = 6
        self.assertEqual( tokens_to_unsigned_int(q, offset, digits), 12345 )


    def test_set_maths_vocabulary(self):
        cfg = MathsConfig()
        set_maths_vocabulary(cfg)

        self.assertEqual( token_to_char(cfg, 4), '4')
        self.assertEqual( token_to_char(cfg, MathsToken.MULT), '*')
        self.assertEqual( tokens_to_string(cfg, [MathsToken.EQUALS,4,0,7]), '=407')
  

    def test_sort_unique_digits(self):
        self.assertEqual( sort_unique_digits("A1231231278321", False), "12378")
        self.assertEqual( sort_unique_digits("A1231231278321", True), "87321")


    # During the construction of the test data, we check that the complexity of the question matches what the test data believes it is
    def test_make_maths_test_questions_and_answers(self):
        cfg = MathsConfig()
        cfg.perc_sub = 60
        cfg.use_cuda = False
        set_maths_vocabulary(cfg)
        
        # Addition questions 
        make_maths_s0_questions_and_answers(cfg)
        make_maths_s1_questions_and_answers(cfg)
        make_maths_s2_questions_and_answers(cfg)
        make_maths_s3_questions_and_answers(cfg)
        make_maths_s4_questions_and_answers(cfg)
        make_maths_s5_questions_and_answers(cfg)
            
        # Subtraction questions with positive (or zero) answers
        make_maths_m0_questions_and_answers(cfg) 
        make_maths_m1_questions_and_answers(cfg)
        make_maths_m2_questions_and_answers(cfg)
        make_maths_m3_questions_and_answers(cfg)
            
        # Subtraction questions with negative answers
        make_maths_n1_questions_and_answers(cfg)         
        make_maths_n2_questions_and_answers(cfg)
        make_maths_n3_questions_and_answers(cfg)
        make_maths_n4_questions_and_answers(cfg)
      
  
    def test_repeat_digit(self):
        cfg = MathsConfig()
        cfg.n_digits = 6
        
        self.assertEqual( cfg.repeat_digit(4), 444444)


if __name__ == '__main__':
    unittest.main()
